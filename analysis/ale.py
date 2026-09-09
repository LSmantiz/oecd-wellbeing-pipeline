"""Accumulated Local Effects (ALE), following iml's implementation.

Ported from R/FeatureEffect-ale.R in the iml package (Molnar & Schratz), which
the original R analysis used via FeatureEffect$new(method = "ale"). The R
source was consulted directly so that the numerical procedure matches rather
than merely the concept.

Correspondence:
    ale_1d  <->  calculate.ale.num
    ale_2d  <->  calculate.ale.num.num

Categorical features (calculate.ale.cat, calculate.ale.num.cat) are not ported:
every predictor in this analysis is numeric.

Two details in the 2D case are easy to get wrong and are reproduced here
deliberately:

  - Empty cells are imputed from their nearest non-empty neighbour in
    normalised grid coordinates, not set to zero. With 388 observations on a
    10x10 grid many cells are empty, so this materially changes the surface.
    iml uses yaImpute::ann for this; a direct nearest-neighbour search is used
    here, which gives the same answer without the dependency.

  - The main effects are removed by accumulating weighted differences along
    each axis (ale1, ale2) and subtracting them along with a global constant
    fJ0, rather than by simple row/column centring. What remains is the
    second-order effect only: the surface is the interaction, not the total
    effect of the pair.

One deliberate deviation: iml derives interval counts with table(), which drops
empty intervals and can misalign the centring weights when a quantile grid
produces one. Counts here are computed with an explicit minimum length, so
empty intervals contribute a weight of zero instead.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ALEResult:
    feature: str
    x: np.ndarray  # grid points (interval edges)
    ale: np.ndarray  # accumulated, centred effect at each grid point
    counts: np.ndarray  # observations per interval


@dataclass
class ALEResult2D:
    features: tuple[str, str]
    x1: np.ndarray
    x2: np.ndarray
    ale: np.ndarray  # shape (len(x1), len(x2))
    counts: np.ndarray  # shape (len(x1) - 1, len(x2) - 1)


def _grid(values: np.ndarray, grid_size: int) -> np.ndarray:
    """Quantile grid, deduplicated, as iml's get.grid does for ALE."""
    probs = np.linspace(0, 1, grid_size + 1)
    edges = np.unique(np.quantile(values, probs))
    if len(edges) < 2:
        raise ValueError("feature has too few distinct values for ALE")
    return edges


def _interval_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """findInterval(values, edges, left.open = TRUE), with 0 clipped to 1.

    Returns 1-based interval numbers: index k means edges[k-1] < v <= edges[k].
    """
    idx = np.searchsorted(edges, values, side="left")
    return np.clip(idx, 1, len(edges) - 1)


def ale_1d(model, X: pd.DataFrame, feature: str, grid_size: int = 20) -> ALEResult:
    """First-order ALE for one numeric feature (iml: calculate.ale.num)."""
    values = X[feature].to_numpy()
    edges = _grid(values, grid_size)
    idx = _interval_index(values, edges)

    lower, upper = X.copy(), X.copy()
    lower[feature] = edges[idx - 1]
    upper[feature] = edges[idx]

    deltas = model.predict(upper) - model.predict(lower)

    n_intervals = len(edges) - 1
    counts = np.bincount(idx - 1, minlength=n_intervals)
    sums = np.bincount(idx - 1, weights=deltas, minlength=n_intervals)

    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    accumulated = np.concatenate([[0.0], np.cumsum(local)])

    # centre on the weighted mean of interval midpoint effects
    midpoints = (accumulated[:-1] + accumulated[1:]) / 2
    fJ0 = np.sum(midpoints * counts) / counts.sum()

    return ALEResult(feature, edges, accumulated - fJ0, counts)


def _impute_empty_cells(
    local: np.ndarray, counts: np.ndarray, e1: np.ndarray, e2: np.ndarray
) -> np.ndarray:
    """Fill empty cells from the nearest occupied cell (iml: impute_cells).

    Distance is measured between cell midpoints, each axis normalised by its
    own range so the two features contribute comparably.
    """
    missing = counts == 0
    if not missing.any() or missing.all():
        return local

    r1 = e1.max() - e1.min()
    r2 = e2.max() - e2.min()
    m1 = (e1[:-1] + e1[1:]) / (2 * r1)
    m2 = (e2[:-1] + e2[1:]) / (2 * r2)

    grid1, grid2 = np.meshgrid(m1, m2, indexing="ij")
    coords = np.column_stack([grid1.ravel(), grid2.ravel()])
    flat_missing = missing.ravel()

    known = coords[~flat_missing]
    unknown = coords[flat_missing]

    # squared distance from every empty cell to every occupied cell
    d2 = ((unknown[:, None, :] - known[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)

    filled = local.ravel().copy()
    filled[flat_missing] = local.ravel()[~flat_missing][nearest]
    return filled.reshape(local.shape)


def ale_2d(
    model, X: pd.DataFrame, features: tuple[str, str], grid_size: int = 10
) -> ALEResult2D:
    """Second-order ALE for a pair of numeric features.

    Follows iml's calculate.ale.num.num. The returned surface is the
    interaction effect: both main effects have been removed.
    """
    f1, f2 = features
    v1, v2 = X[f1].to_numpy(), X[f2].to_numpy()

    e1, e2 = _grid(v1, grid_size), _grid(v2, grid_size)
    i1, i2 = _interval_index(v1, e1), _interval_index(v2, e2)
    n1, n2 = len(e1) - 1, len(e2) - 1

    def predict_at(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        Z = X.copy()
        Z[f1] = a
        Z[f2] = b
        return model.predict(Z)

    # second-order difference over each cell's four corners
    p11 = predict_at(e1[i1 - 1], e2[i2 - 1])
    p21 = predict_at(e1[i1], e2[i2 - 1])
    p12 = predict_at(e1[i1 - 1], e2[i2])
    p22 = predict_at(e1[i1], e2[i2])
    delta = (p22 - p21) - (p12 - p11)

    cell = (i1 - 1) * n2 + (i2 - 1)
    counts = np.bincount(cell, minlength=n1 * n2).reshape(n1, n2)
    sums = np.bincount(cell, weights=delta, minlength=n1 * n2).reshape(n1, n2)

    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    local = _impute_empty_cells(local, counts, e1, e2)

    # accumulate along both axes, with a zero row and column at the origin
    acc = np.zeros((n1 + 1, n2 + 1))
    acc[1:, 1:] = np.cumsum(np.cumsum(local, axis=0), axis=1)

    # --- main effect of feature 1 -------------------------------------------
    d1 = np.diff(acc, axis=0)  # (n1, n2 + 1)
    pair1 = (d1[:, :-1] + d1[:, 1:]) / 2  # (n1, n2)
    w1 = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        local1 = np.where(w1 > 0, (pair1 * counts).sum(axis=1) / np.maximum(w1, 1), 0.0)
    ale1 = np.concatenate([[0.0], np.cumsum(local1)])  # (n1 + 1,)

    # --- main effect of feature 2 -------------------------------------------
    d2 = np.diff(acc, axis=1)  # (n1 + 1, n2)
    pair2 = (d2[:-1, :] + d2[1:, :]) / 2  # (n1, n2)
    w2 = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        local2 = np.where(w2 > 0, (pair2 * counts).sum(axis=0) / np.maximum(w2, 1), 0.0)
    ale2 = np.concatenate([[0.0], np.cumsum(local2)])  # (n2 + 1,)

    # --- remove main effects and centre --------------------------------------
    dd = acc - ale1[:, None] - ale2[None, :]
    corners = (dd[:-1, :-1] + dd[:-1, 1:] + dd[1:, :-1] + dd[1:, 1:]) / 4
    fJ0 = (counts * corners).sum() / counts.sum()

    return ALEResult2D(features, e1, e2, dd - fJ0, counts)


# ------------------------------------------------------------------ plotting


def plot_ale_1d(result: ALEResult, ax=None, label_width: int = 30):
    """Step plot of a first-order ALE curve, with a data-density rug."""
    import matplotlib.pyplot as plt

    from analysis.labels import label

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    ax.step(result.x, result.ale, where="post", color="black", linewidth=1.5)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)

    midpoints = (result.x[:-1] + result.x[1:]) / 2
    ax.scatter(
        midpoints,
        np.zeros_like(midpoints),
        marker="|",
        s=8 + 40 * result.counts / max(result.counts.max(), 1),
        color="grey",
        alpha=0.5,
    )

    ax.set_xlabel(label(result.feature, label_width), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_ale_2d(result: ALEResult2D, ax=None, label_width: int = 30, cmap="Greys"):
    """Heat map of a second-order ALE surface."""
    import matplotlib.pyplot as plt

    from analysis.labels import label

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    f1, f2 = result.features
    mesh = ax.pcolormesh(result.x1, result.x2, result.ale.T, cmap=cmap, shading="auto")
    plt.colorbar(mesh, ax=ax, label="ALE")

    ax.set_xlabel(label(f1, label_width))
    ax.set_ylabel(label(f2, label_width))
    ax.spines[["top", "right"]].set_visible(False)
    return ax