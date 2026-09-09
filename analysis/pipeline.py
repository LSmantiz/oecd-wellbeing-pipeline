"""The complete analysis for one dataset.

Runs the same sequence as the main Quarto document — predictor selection,
hyperparameter tuning, model comparison, importance, ALE, interactions — and
returns every artifact as a plain data structure so it can be cached to disk and
compared across points on the completeness frontier.

Nothing here plots. Rendering is the document's job; this module only computes.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 4595
TARGET = "SUBJ_LIFE_SAT"


@dataclass
class AnalysisResult:
    n_dropped: int
    n_regions: int
    n_indicators: int

    selection_path: pd.DataFrame
    selected: list[str]

    best_params_rf: dict
    best_params_xgb: dict
    metrics: pd.DataFrame

    importance: pd.DataFrame
    ale_1d: pd.DataFrame
    interactions: pd.DataFrame
    interactions_top: pd.DataFrame

    top_feature: str
    top_partner: str
    ale_2d_x1: np.ndarray = field(default_factory=lambda: np.array([]))
    ale_2d_x2: np.ndarray = field(default_factory=lambda: np.array([]))
    ale_2d_values: np.ndarray = field(default_factory=lambda: np.array([]))
    ale_2d_counts: np.ndarray = field(default_factory=lambda: np.array([]))


def run_analysis(
    wide: pd.DataFrame,
    n_dropped: int = 0,
    seed: int = SEED,
    tune_size: int = 60,
    n_repeats: int = 100,
    ale_grid: int = 20,
    ale_2d_grid: int = 20,
    interaction_grid: int = 30,
    with_xgboost: bool = True,
    verbose: bool = True,
) -> AnalysisResult:
    """Run selection, tuning, comparison and interpretation on one dataset."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from analysis.ale import ale_1d, ale_2d
    from analysis.importance import feature_importance
    from analysis.interactions import interaction_strength, interaction_with
    from analysis.select_features import select_features, choose_configuration    
    from analysis.tune import stratified_split, tune_forest, tune_xgboost

    def log(msg: str) -> None:
        if verbose:
            print(f"    {msg}", flush=True)

    # --- predictor selection ------------------------------------------------
    log("selecting predictors")
    path = select_features(wide, seed=seed)
    best = choose_configuration(path, rule="1se")
    selected = list(best.variables)
    log(f"{len(selected)} predictors, OOB {best.oob_mse:.4f}")

    # --- tuning -------------------------------------------------------------
    log("tuning random forest")
    split = stratified_split(wide, predictors=selected, seed=seed)
    rf_search = tune_forest(split, size=tune_size, seed=seed)

    log("tuning xgboost")
    xgb_search = tune_xgboost(split, size=tune_size, seed=seed)

    lm = LinearRegression().fit(split.X_train, split.y_train)

    models = {"rf": rf_search.best_estimator_, "lm": lm}
    best_params_xgb: dict = {}

    if with_xgboost:
        log("tuning xgboost")
        xgb_search = tune_xgboost(split, size=tune_size, seed=seed)
        models["xgb"] = xgb_search.best_estimator_
        best_params_xgb = dict(xgb_search.best_params_)

    rows = []
    for name, model in models.items():
        pred = model.predict(split.X_test)
        rows.append(
            {
                "model": name,
                "rmse": float(np.sqrt(mean_squared_error(split.y_test, pred))),
                "rsq": float(r2_score(split.y_test, pred)),
                "mae": float(mean_absolute_error(split.y_test, pred)),
            }
        )
    metrics = pd.DataFrame(rows)

    # --- interpretation, on the random forest -------------------------------
    X_all, y_all = wide[selected], wide[TARGET]
    rf = models["rf"]

    log("permutation importance")
    importance = feature_importance(rf, X_all, y_all, n_repeats=n_repeats, seed=seed)

    log("ale curves")
    ale_rows = []
    for feature in importance.feature:
        r = ale_1d(rf, X_all, feature, grid_size=ale_grid)
        ale_rows.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "x": r.x,
                    "ale": r.ale,
                    "count": np.concatenate([[0], r.counts]),
                }
            )
        )
    ale_frame = pd.concat(ale_rows, ignore_index=True)

    log("interaction strengths")
    interactions = interaction_strength(
        rf, X_all, n_grid=interaction_grid, n_background=interaction_grid, seed=seed
    )
    top_feature = interactions.iloc[0].feature

    log(f"pairwise interactions with {top_feature}")
    interactions_top = interaction_with(
        rf,
        X_all,
        top_feature,
        n_grid=interaction_grid,
        n_background=interaction_grid,
        seed=seed,
    )
    top_partner = interactions_top.iloc[0].feature

    log(f"2d ale for {top_feature} x {top_partner}")
    surface = ale_2d(rf, X_all, (top_feature, top_partner), grid_size=ale_2d_grid)

    return AnalysisResult(
        n_dropped=n_dropped,
        n_regions=int(wide.shape[0]),
        n_indicators=int(wide.shape[1] - 1),
        selection_path=path.drop(columns="variables"),
        selected=selected,
        best_params_rf=dict(rf_search.best_params_),
        best_params_xgb=best_params_xgb,
        metrics=metrics,
        importance=importance,
        ale_1d=ale_frame,
        interactions=interactions,
        interactions_top=interactions_top,
        top_feature=top_feature,
        top_partner=top_partner,
        ale_2d_x1=surface.x1,
        ale_2d_x2=surface.x2,
        ale_2d_values=surface.ale,
        ale_2d_counts=surface.counts,
    )


# ---------------------------------------------------------------- persistence


def save_result(result: AnalysisResult, directory: Path) -> None:
    """Write every artifact for one step to its own directory."""
    import json

    directory.mkdir(parents=True, exist_ok=True)

    result.selection_path.to_csv(directory / "selection_path.csv", index=False)
    result.metrics.to_csv(directory / "metrics.csv", index=False)
    result.importance.to_csv(directory / "importance.csv", index=False)
    result.ale_1d.to_csv(directory / "ale_1d.csv", index=False)
    result.interactions.to_csv(directory / "interactions.csv", index=False)
    result.interactions_top.to_csv(directory / "interactions_top.csv", index=False)

    np.savez(
        directory / "ale_2d.npz",
        x1=result.ale_2d_x1,
        x2=result.ale_2d_x2,
        values=result.ale_2d_values,
        counts=result.ale_2d_counts,
    )

    meta = {
        "n_dropped": result.n_dropped,
        "n_regions": result.n_regions,
        "n_indicators": result.n_indicators,
        "selected": result.selected,
        "best_params_rf": {k: _plain(v) for k, v in result.best_params_rf.items()},
        "best_params_xgb": {k: _plain(v) for k, v in result.best_params_xgb.items()},
        "top_feature": result.top_feature,
        "top_partner": result.top_partner,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def _plain(value):
    """numpy scalars are not JSON-serialisable."""
    return value.item() if hasattr(value, "item") else value


def load_result(directory: Path) -> dict:
    """Read one step's cached artifacts back as a dict."""
    import json

    surface = np.load(directory / "ale_2d.npz")

    return {
        **json.loads((directory / "meta.json").read_text()),
        "selection_path": pd.read_csv(directory / "selection_path.csv"),
        "metrics": pd.read_csv(directory / "metrics.csv"),
        "importance": pd.read_csv(directory / "importance.csv"),
        "ale_1d": pd.read_csv(directory / "ale_1d.csv"),
        "interactions": pd.read_csv(directory / "interactions.csv"),
        "interactions_top": pd.read_csv(directory / "interactions_top.csv"),
        "ale_2d": {k: surface[k] for k in surface.files},
    }


def load_all(root: Path) -> dict[int, dict]:
    """Every cached step, keyed by n_dropped."""
    out = {}
    for directory in sorted(root.glob("step_*")):
        if (directory / "meta.json").exists():
            result = load_result(directory)
            out[result["n_dropped"]] = result
    return dict(sorted(out.items()))
