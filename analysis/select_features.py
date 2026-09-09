"""Recursive feature elimination by random forest importance.

Python translation of varSelRanger (Simon Larsen), as used in the original R
analysis: fit a forest, record the out-of-bag error, drop the least important
`frac` of predictors, repeat until `min_vars` remain.

Matching ranger's defaults matters here:
    ranger  mtry           = floor(sqrt(p))   -> max_features="sqrt"
    ranger  min.node.size  = 5 (regression)   -> min_samples_leaf=5
    ranger  num.trees      = 500              -> n_estimators=500
    ranger  importance     = "impurity"       -> feature_importances_ (MDI)
    ranger  prediction.error (regression)     = OOB MSE, computed here from
                                                oob_prediction_
sklearn's own oob_score_ is R-squared, not MSE, so it is not used.

On choosing a configuration: the error curve is flat over a broad range, so
taking the strict minimum is unstable - in this dataset 16 and 21 predictors
differ by 0.0006 in OOB MSE, and which one wins changes with the random seed.
The original R analysis handled this by judgement, choosing 16 predictors
because it "does not significantly increase the OOB but reduces the amount of
variables". `choose_configuration` makes that judgement into a rule: take the
fewest predictors whose error is within one standard error of the minimum.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SEED = 4595
TARGET = "SUBJ_LIFE_SAT"


def fit_forest(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> RandomForestRegressor:
    """Fit a forest with ranger-equivalent hyperparameters."""
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        oob_score=True,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def oob_error(rf: RandomForestRegressor, y: pd.Series) -> tuple[float, float]:
    """Out-of-bag MSE and its standard error.

    The MSE is ranger's prediction.error. The standard error is that of the mean
    squared residual across observations, which is what makes a one-standard-
    error selection rule possible from a single fit.
    """
    residuals = (y.to_numpy() - rf.oob_prediction_) ** 2
    mse = float(residuals.mean())
    se = float(residuals.std(ddof=1) / np.sqrt(len(residuals)))
    return mse, se


def select_features(
    data: pd.DataFrame,
    target: str = TARGET,
    frac: float = 0.2,
    min_vars: int = 1,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the elimination and return one row per step.

    Columns: step, n_variables, oob_mse, oob_se, variables.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be strictly between 0 and 1")
    if target not in data.columns:
        raise KeyError(f"{target} not in data")

    y = data[target]
    variables = [c for c in data.columns if c != target]

    rows = []
    step = 0

    while len(variables) >= min_vars:
        step += 1
        rf = fit_forest(data[variables], y, seed=seed)
        mse, se = oob_error(rf, y)

        rows.append(
            {
                "step": step,
                "n_variables": len(variables),
                "oob_mse": mse,
                "oob_se": se,
                "variables": list(variables),
            }
        )

        importance = pd.Series(rf.feature_importances_, index=variables)
        n_keep = int(np.floor(len(importance) * (1 - frac)))
        variables = list(importance.sort_values(ascending=False).head(n_keep).index)

    return pd.DataFrame(rows)


def choose_configuration(path: pd.DataFrame, rule: str = "1se") -> pd.Series:
    """Pick one configuration from the elimination path.

    "1se"     the fewest predictors whose OOB error is within one standard
              error of the minimum. Preferred: the error curve is flat across a
              wide range, so the strict minimum moves with the seed while this
              rule does not.
    "min"     the strict minimum, for comparison.
    """
    if rule == "min":
        return path.loc[path.oob_mse.idxmin()]

    if rule != "1se":
        raise ValueError("rule must be '1se' or 'min'")

    best = path.loc[path.oob_mse.idxmin()]
    threshold = best.oob_mse + best.oob_se

    within = path[path.oob_mse <= threshold]
    return within.loc[within.n_variables.idxmin()]


def load_analysis_data(warehouse: Path | str, scaled: bool = False) -> pd.DataFrame:
    """Read the analysis mart and pivot to wide, one row per region.

    Unscaled by default, so that standardisation can happen inside a modelling
    pipeline rather than being baked into the input. Note that the OOB MSE
    depends on the scale of the target, so figures from scaled and unscaled runs
    are not comparable.
    """
    import duckdb

    model = "mart_analysis_scaled" if scaled else "mart_analysis_long"
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        long = con.sql(f"select reg_id, indicator, value from {model}").df()
    finally:
        con.close()

    return (
        long.pivot(index="reg_id", columns="indicator", values="value")
        .rename_axis(columns=None)
        .sort_index()
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    wide = load_analysis_data(root / "warehouse.duckdb")
    print(f"data: {wide.shape[0]} regions, {wide.shape[1] - 1} predictors")

    path = select_features(wide)
    print(path[["n_variables", "oob_mse", "oob_se"]].to_string(index=False))

    strict = choose_configuration(path, rule="min")
    parsimonious = choose_configuration(path, rule="1se")

    print(f"\nminimum:      {int(strict.n_variables):>3} predictors, "
          f"OOB {strict.oob_mse:.4f} (SE {strict.oob_se:.4f})")
    print(f"within 1 SE:  {int(parsimonious.n_variables):>3} predictors, "
          f"OOB {parsimonious.oob_mse:.4f}")

    print("\nselected:")
    for v in sorted(parsimonious.variables):
        print(f"  {v}")