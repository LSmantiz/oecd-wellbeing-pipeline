# src/reshape.py
"""Reshape long OECD source data into wide per-region tables."""

import pandas as pd
from config import MIN_YEAR


def keep_latest(df: pd.DataFrame, group_cols: list[str], time_col: str = "TIME") -> pd.DataFrame:
    """Keep rows from the most recent year available within each group."""
    latest = df.groupby(group_cols)[time_col].transform("max")
    return df[df[time_col] == latest]


def prepare_long(
    df: pd.DataFrame,
    regions: list[str],
    key_cols: list[str],
    min_year: int = MIN_YEAR,
    extra_filters: dict[str, str] | None = None,
    whitelist: tuple[str, list[str]] | None = None,
    dedupe_keys: bool = False,
) -> pd.DataFrame:
    """Subset to matching regions and the latest observation per series."""
    out = df[df["REG_ID"].isin(regions)]
    out = out[out["POS"] == "ALL"]

    for col, val in (extra_filters or {}).items():
        out = out[out[col] == val]

    if whitelist:
        col, values = whitelist
        out = out[out[col].isin(values)]

    out = out[out["TIME"] >= min_year]

    keys = ["REG_ID", "Region"] + key_cols
    out = keep_latest(out, keys)

    if dedupe_keys:
        # social: duplicate rows carry identical Values (verified against R)
        return out.drop_duplicates(subset=keys, keep="first")
    return out.drop_duplicates(subset=keys + ["Value"])


def pivot_wide(df: pd.DataFrame, key_cols: list[str], value_col: str = "Value") -> pd.DataFrame:
    """Widen long data into one column per key combination (e.g. VAR_SEX)."""
    key = df[key_cols].astype(str).agg("_".join, axis=1)
    return (
        df.assign(_key=key)
        .pivot(index=["REG_ID", "Region"], columns="_key", values=value_col)
        .reset_index()
        .rename_axis(columns=None)
    )