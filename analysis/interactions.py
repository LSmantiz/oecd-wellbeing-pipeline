"""Friedman's H-statistic for feature interactions.

Python equivalent of iml::Interaction, which has no counterpart in scikit-learn.

Two quantities, both built from centred partial dependence functions:

  overall (interaction_strength)
      H_j^2 = sum_i [ f(x_i) - PD_j(x_ij) - PD_-j(x_i,-j) ]^2 / sum_i f(x_i)^2
      "how much of the prediction is NOT explained by feature j and everything
      else acting separately"

  pairwise (interaction_with)
      H_jk^2 = sum_i [ PD_jk - PD_j - PD_k ]^2 / sum_i PD_jk^2
      "how much of the joint effect of j and k is NOT the sum of their
      individual effects"

H is reported as the square root, so it is on a 0-1 scale where 0 means no
interaction. Values can exceed 1 when the denominator is small - iml has the
same behaviour - so treat large values on weak features with suspicion.

Cost: partial dependence is estimated by Monte Carlo, so the number of model
evaluations grows as n_grid * n_background per feature. Both default to 30,
matching iml's grid.size, which keeps a 16-feature model tractable. Raise them
for a more stable estimate at proportionally more compute.

Note that the H-statistic is a variance-based measure computed from the fitted
model, not a test: a tree ensemble will report nonzero interaction even for
data generated additively, because the trees themselves introduce it. Compare
across models rather than reading a single number as evidence.
"""

from typing import Sequence

import numpy as np
import pandas as pd

SEED = 4595


def _centre(v: np.ndarray) -> np.ndarray:
    return v - v.mean()


def _partial_dependence(
    model,
    background: pd.DataFrame,
    eval_points: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """PD of `features`, evaluated at each row of `eval_points`.

    For every evaluation point the named features are held at that point's
    values while all other features are taken from the background sample; the
    prediction is averaged over the background.
    """
    features = list(features)
    n_eval, n_bg = len(eval_points), len(background)

    tiled = pd.concat([background] * n_eval, ignore_index=True)
    values = eval_points[features].to_numpy()
    tiled[features] = np.repeat(values, n_bg, axis=0)

    preds = np.asarray(model.predict(tiled), dtype=float)
    return preds.reshape(n_eval, n_bg).mean(axis=1)


def _sample(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return X if len(X) <= n else X.sample(n=n, random_state=seed)


def interaction_strength(
    model,
    X: pd.DataFrame,
    features: Sequence[str] | None = None,
    n_grid: int = 30,
    n_background: int = 30,
    seed: int = SEED,
) -> pd.DataFrame:
    """Overall interaction strength for each feature (iml: Interaction$new)."""
    features = list(features) if features is not None else list(X.columns)

    eval_points = _sample(X, n_grid, seed)
    background = _sample(X, n_background, seed + 1)

    f = _centre(np.asarray(model.predict(eval_points), dtype=float))
    denominator = np.sum(f**2)

    rows = []
    for feature in features:
        others = [c for c in X.columns if c != feature]

        pd_j = _centre(_partial_dependence(model, background, eval_points, [feature]))
        pd_not_j = _centre(_partial_dependence(model, background, eval_points, others))

        numerator = np.sum((f - pd_j - pd_not_j) ** 2)
        h = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
        rows.append({"feature": feature, "interaction": h})

    return (
        pd.DataFrame(rows).sort_values("interaction", ascending=False).reset_index(drop=True)
    )


def interaction_with(
    model,
    X: pd.DataFrame,
    feature: str,
    others: Sequence[str] | None = None,
    n_grid: int = 30,
    n_background: int = 30,
    seed: int = SEED,
) -> pd.DataFrame:
    """Pairwise interaction strength between `feature` and each other feature.

    Equivalent to iml's Interaction$new(predictor, feature = <name>).
    """
    others = list(others) if others is not None else [c for c in X.columns if c != feature]

    eval_points = _sample(X, n_grid, seed)
    background = _sample(X, n_background, seed + 1)

    pd_j = _centre(_partial_dependence(model, background, eval_points, [feature]))

    rows = []
    for other in others:
        if other == feature:
            continue
        pd_k = _centre(_partial_dependence(model, background, eval_points, [other]))
        pd_jk = _centre(
            _partial_dependence(model, background, eval_points, [feature, other])
        )

        denominator = np.sum(pd_jk**2)
        numerator = np.sum((pd_jk - pd_j - pd_k) ** 2)
        h = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
        rows.append({"feature": other, "interaction": h})

    return (
        pd.DataFrame(rows).sort_values("interaction", ascending=False).reset_index(drop=True)
    )


def plot_interaction(
    result: pd.DataFrame,
    ax=None,
    title: str | None = None,
    xlabel: str = "Overall interaction strength",
    label_width: int = 30,
):
    """Horizontal bar plot of interaction strengths."""
    import matplotlib.pyplot as plt

    from analysis.labels import label

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    d = result.iloc[::-1]
    positions = np.arange(len(d))

    ax.barh(positions, d.interaction, color="grey", edgecolor="black", height=0.6)
    ax.set_yticks(positions)
    ax.set_yticklabels([label(f, label_width) for f in d.feature], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)

    return ax