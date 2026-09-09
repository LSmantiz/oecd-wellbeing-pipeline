"""Reading points on the completeness frontier.

The analysis dataset requires every indicator to be observed for every region,
so a single poorly-covered region removes an indicator for all of them.
Dropping the worst-covered regions therefore buys indicators back.

Rather than materialising one stored dataset per step - which would hold the
same values many times over - the filtering happens at read time against two
small dbt models:

    mart_region_completeness  one row per region, with its drop order
    mart_indicators_resolved  every region-indicator value, before completeness

A consequence worth having: n_dropped is a query parameter, not a fixed
materialisation, so the frontier can be sampled at any resolution without
rebuilding anything.

Connections are opened read-only. DuckDB permits many readers or one writer, so
an interactive session holding a write lock will otherwise block these reads.
"""

from pathlib import Path

import pandas as pd

TARGET = "SUBJ_LIFE_SAT"


def load_frontier_step(warehouse: Path | str, n_dropped: int = 0) -> pd.DataFrame:
    """Drop the n worst-covered regions and return the resulting wide dataset.

    Only indicators observed for every retained region are kept, so the result
    is rectangular and complete. n_dropped = 0 reproduces mart_analysis_long.
    """
    import duckdb

    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        long = con.sql(
            f"""
            with kept as (
                select reg_id
                from mart_region_completeness
                where drop_rank > {int(n_dropped)}
            ),

            restricted as (
                select b.reg_id, b.indicator, b.value
                from mart_indicators_resolved b
                join kept k using (reg_id)
            ),

            complete as (
                select indicator
                from restricted
                group by indicator
                having count(distinct reg_id) = (select count(*) from kept)
            )

            select r.reg_id, r.indicator, r.value
            from restricted r
            join complete c using (indicator)
            """
        ).df()
    finally:
        con.close()

    if long.empty:
        raise ValueError(f"no rows for n_dropped = {n_dropped}")

    return (
        long.pivot(index="reg_id", columns="indicator", values="value")
        .rename_axis(columns=None)
        .sort_index()
    )


def load_frontier_summary(warehouse: Path | str) -> pd.DataFrame:
    """The regions/indicators trade-off table from dbt."""
    import duckdb

    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        return con.sql(
            "select * from mart_completeness_frontier order by n_dropped"
        ).df()
    finally:
        con.close()


def available_steps(warehouse: Path | str) -> list[int]:
    """Step sizes defined by the dbt frontier summary."""
    return load_frontier_summary(warehouse).n_dropped.astype(int).tolist()


def common_regions(warehouse: Path | str, max_dropped: int) -> set[str]:
    """Regions surviving every step up to max_dropped.

    Useful when comparing model performance across the frontier: the retained
    sample changes at each step, so a metric computed on whatever remains is
    not measured on the same thing twice.
    """
    import duckdb

    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        out = con.sql(
            f"""
            select reg_id
            from mart_region_completeness
            where drop_rank > {int(max_dropped)}
            """
        ).df()
    finally:
        con.close()
    return set(out.reg_id)


def dropped_regions(warehouse: Path | str, n_dropped: int) -> pd.DataFrame:
    """Which regions are removed at a given step, and how much they were missing.

    Regions are dropped worst-first, and poor coverage is unlikely to be random,
    so it is worth inspecting which parts of the sample the frontier trades away.
    """
    import duckdb

    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        return con.sql(
            f"""
            select reg_id, n_present, n_missing, drop_rank
            from mart_region_completeness
            where drop_rank <= {int(n_dropped)}
            order by drop_rank
            """
        ).df()
    finally:
        con.close()