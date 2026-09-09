"""Model-agnostic interpretation: permutation feature importance.

Translation of the iml::FeatureImp block. Two differences from a naive use of
sklearn's permutation_importance are worth knowing:

  - iml reports importance as a RATIO by default: the permuted loss divided by
    the original loss, so 1.0 means "permuting this feature changes nothing".
    sklearn reports the drop in score. The ratio is reconstructed here so the
    plots are on the same scale as the original figures.

  - the original computes importance on the full dataset (all 388 regions),
    not on the held-out test set. That is reproduced here for comparability,
    but note the consequence: importance is measured partly on data the model
    was trained on, so values are optimistic. `data="test"` gives the
    out-of-sample version, which is the more defensible choice for a new
    analysis.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

SEED = 4595


def feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 100,
    seed: int = SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Permutation importance as an MSE ratio, matching iml::FeatureImp.

    Returns one row per feature with the median ratio and the 5th/95th
    percentiles across repetitions, sorted most important first.
    """
    baseline = mean_squared_error(y, model.predict(X))

    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_squared_error",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )

    # importances_ holds (baseline_score - permuted_score) per repetition, with
    # scores negated by sklearn's convention, so permuted MSE = baseline + value
    permuted_mse = baseline + result.importances

    ratios = permuted_mse / baseline

    out = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": np.median(ratios, axis=1),
            "lower": np.percentile(ratios, 5, axis=1),
            "upper": np.percentile(ratios, 95, axis=1),
        }
    )
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importance(
    importance: pd.DataFrame,
    ax=None,
    title: str | None = None,
    label_width: int = 30,
    highlight_demographic: bool = True,
):
    """Horizontal importance plot with 5-95 percentile whiskers."""
    import matplotlib.pyplot as plt

    from analysis.labels import is_demographic, label

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    d = importance.iloc[::-1]  # most important at the top
    positions = np.arange(len(d))

    ax.hlines(positions, d.lower, d.upper, color="grey", linewidth=1.5)
    ax.scatter(d.importance, positions, color="black", s=30, zorder=3)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)

    ax.set_yticks(positions)
    ax.set_yticklabels([label(f, label_width) for f in d.feature], fontsize=8)

    if highlight_demographic:
        for tick, feature in zip(ax.get_yticklabels(), d.feature):
            if is_demographic(feature):
                tick.set_fontweight("bold")

    ax.set_xlabel("Predictor importance (permuted MSE / original MSE)")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)

    return ax