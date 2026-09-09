"""Hyperparameter tuning for the random forest and XGBoost models.

The search machinery is model-agnostic: a parameter specification describes the
bounds and scale of each hyperparameter, a Latin hypercube spreads candidate
points through that space, and the candidates are evaluated by k-fold CV.

Two things have no direct sklearn equivalent and are reimplemented:

  - rsample::initial_split(strata = <numeric>) bins a continuous outcome into
    quantiles and samples within each. sklearn stratifies on discrete labels
    only, so the binning is done explicitly in `stratified_split`.

  - dials::grid_space_filling / grid_max_entropy spread candidate points evenly
    through the parameter space. scipy's Latin hypercube does the same job;
    RandomizedSearchCV's uniform draw would cluster more.

Parameter ranges follow the dials defaults used in the original script.
Note that learn_rate and loss_reduction are sampled on a log10 scale, as dials
does: sampling them linearly would place almost every candidate at the top of
the range.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

SEED = 4595
TARGET = "SUBJ_LIFE_SAT"

Scale = Literal["linear", "log10"]


@dataclass
class Param:
    """One tunable hyperparameter: bounds, sampling scale, and output type."""

    low: float
    high: float
    scale: Scale = "linear"
    integer: bool = True

    def transform(self, unit_value: float) -> Any:
        raw = self.low + unit_value * (self.high - self.low)
        value = 10.0**raw if self.scale == "log10" else raw
        return int(round(value)) if self.integer else float(value)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def stratified_split(
    data: pd.DataFrame,
    predictors: Sequence[str],
    target: str = TARGET,
    train_size: float = 0.75,
    n_strata: int = 4,
    seed: int = SEED,
) -> Split:
    """Train/test split stratified on quantile bins of a continuous outcome."""
    y = data[target]
    strata = pd.qcut(y, q=n_strata, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        data[list(predictors)],
        y,
        train_size=train_size,
        stratify=strata,
        random_state=seed,
    )
    return Split(X_train, X_test, y_train, y_test)


def space_filling_grid(
    spec: dict[str, Param], size: int = 60, seed: int = SEED
) -> list[dict[str, list]]:
    """Latin hypercube sample over the given parameter space.

    Returns a list of single-point grids, which is how GridSearchCV accepts a
    set of specific candidates rather than a cartesian product.
    """
    names = list(spec)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=size)

    grid = []
    for row in unit:
        grid.append({name: [spec[name].transform(u)] for name, u in zip(names, row)})
    return grid


# ---------------------------------------------------------------- specifications


def forest_spec(n_predictors: int) -> dict[str, Param]:
    """dials: trees(), min_n(), finalize(mtry(), train)."""
    return {
        "n_estimators": Param(1, 2000),
        "min_samples_leaf": Param(2, 40),
        "max_features": Param(1, n_predictors),
    }


def xgboost_spec() -> dict[str, Param]:
    """dials: min_n(), tree_depth(), learn_rate(), loss_reduction().

    parsnip maps these onto xgboost as:
        min_n          -> min_child_weight
        tree_depth     -> max_depth
        learn_rate     -> learning_rate    (log10 scale)
        loss_reduction -> gamma            (log10 scale)
    """
    return {
        "min_child_weight": Param(2, 40),
        "max_depth": Param(1, 15),
        "learning_rate": Param(-10, -1, scale="log10", integer=False),
        "gamma": Param(-10, 1.5, scale="log10", integer=False),
    }


# ---------------------------------------------------------------- tuning


def tune_model(
    estimator,
    split: Split,
    grid: list[dict[str, list]],
    n_folds: int = 5,
    seed: int = SEED,
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Tune by k-fold CV, selecting on RMSE as the original does."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "rsq": "r2",
            "mae": "neg_mean_absolute_error",
        },
        refit="rmse",
        n_jobs=n_jobs,
        verbose=verbose,
    )
    search.fit(split.X_train, split.y_train)
    return search


def tune_forest(
    split: Split, size: int = 60, seed: int = SEED, n_jobs: int = -1
) -> GridSearchCV:
    grid = space_filling_grid(forest_spec(split.X_train.shape[1]), size=size, seed=seed)
    estimator = RandomForestRegressor(random_state=seed, n_jobs=1)
    return tune_model(estimator, split, grid, seed=seed, n_jobs=n_jobs)


def tune_xgboost(
    split: Split,
    size: int = 60,
    n_estimators: int = 1000,
    seed: int = SEED,
    n_jobs: int = -1,
) -> GridSearchCV:
    """The original fixes trees = 1000 for XGBoost and tunes the rest."""
    from xgboost import XGBRegressor

    grid = space_filling_grid(xgboost_spec(), size=size, seed=seed)
    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    return tune_model(estimator, split, grid, seed=seed, n_jobs=n_jobs)


# ---------------------------------------------------------------- reporting


def tuning_results(search: GridSearchCV) -> pd.DataFrame:
    """Candidate configurations ranked by cross-validated RMSE."""
    cv = pd.DataFrame(search.cv_results_)

    param_cols = [c for c in cv.columns if c.startswith("param_")]
    out = cv[param_cols + ["mean_test_rmse", "mean_test_rsq", "mean_test_mae"]].copy()

    # sklearn reports negated losses so that "greater is better" holds
    out["mean_test_rmse"] = -out["mean_test_rmse"]
    out["mean_test_mae"] = -out["mean_test_mae"]

    out.columns = [
        c.replace("param_", "").replace("mean_test_", "") for c in out.columns
    ]
    return out.sort_values("rmse").reset_index(drop=True)