"""Merge the source frames, apply data corrections, and produce the final datasets."""

import pandas as pd
from pathlib import Path


def merge_all(wellbeing: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join every source frame onto the well-being table.

    Joins on REG_ID alone. Region labels differ between source files for some
    regions (different spellings and language variants for the same REG_ID), so
    joining on ["REG_ID", "Region"] silently drops around 20 regions' worth of
    data. REG_ID is the only reliable key.

    Columns already present in the accumulated frame are dropped from the
    incoming frame, so the well-being version wins for indicators that appear
    in more than one source (AIR_POL, BB_ACC, HOMIC_RA, ROOMS_PC).
    """
    out = wellbeing
    for name, df in frames.items():
        df = df.drop(columns=["Region"], errors="ignore")

        dupes = (set(df.columns) & set(out.columns)) - {"REG_ID"}
        if dupes:
            print(f"{name}: dropping {sorted(dupes)} (already present)")
            df = df.drop(columns=list(dupes))

        before = len(out)
        out = out.merge(df, on="REG_ID", how="left", validate="one_to_one")
        assert len(out) == before, f"{name} join changed row count"

    return out


def correct_de7_density(df: pd.DataFrame, col: str = "POP_DEN_GR_T") -> pd.DataFrame:
    """Correct DE7's population density growth index.

    DE7's 2001 surface area is wrong by an order of magnitude in the source
    data, which inflates the 2001-base density growth index by 100x. Verified
    by recomputing from the 2002 surface (21114.2 km2): the corrected value
    equals the database value / 100.
    """
    if col not in df.columns:
        raise KeyError(
            f"{col} not in data; columns starting POP_DEN: "
            f"{[c for c in df.columns if c.startswith('POP_DEN')]}"
        )

    out = df.copy()
    mask = out["REG_ID"] == "DE7"
    out.loc[mask, col] = out.loc[mask, col] / 100
    return out


def drop_incomplete_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns with no missing values (R's select_if(~all(!is.na(.))))."""
    return df.loc[:, df.notna().all()]


def scale2(df: pd.DataFrame, exclude: tuple[str, ...] = ("REG_ID",)) -> pd.DataFrame:
    """Standardise numeric columns using the sample SD (ddof=1), matching R's sd().

    Note this differs from sklearn's StandardScaler, which uses ddof=0.
    """
    out = df.copy()
    cols = [c for c in out.select_dtypes("number").columns if c not in exclude]
    out[cols] = (out[cols] - out[cols].mean()) / out[cols].std(ddof=1)
    return out


def na_counts_by_row(df: pd.DataFrame) -> pd.Series:
    """Rows ordered by how many missing values they contain, worst first."""
    return df.isna().sum(axis=1).sort_values(ascending=False)


def build_appendix(
    data: pd.DataFrame,
    out_dir: Path,
    steps=range(10, 151, 10),
) -> pd.DataFrame:
    """Write one dataset per number of dropped regions, worst-missing first.

    Regions are dropped in order of how much data they are missing, then
    incomplete columns are removed and the remainder scaled. Scaling happens
    per dataset, so the outputs are not on a common scale.

    Returns a table of the resulting dimensions. Once still_missing reaches 0,
    further steps drop complete regions without recovering any variables.
    """
    ranked = na_counts_by_row(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in steps:
        kept = data.drop(index=ranked.index[:i])
        remaining_incomplete = int((kept.isna().sum(axis=1) > 0).sum())

        kept = scale2(drop_incomplete_columns(kept))
        kept.to_parquet(out_dir / f"data_{i}.parquet", index=False)

        rows.append({
            "erased_reg": i,
            "regions": len(kept),
            "variables": kept.shape[1],
            "still_missing": remaining_incomplete,
        })

    return pd.DataFrame(rows)