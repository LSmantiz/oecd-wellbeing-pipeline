"""Run the full analysis at every point on the completeness frontier.

This is a batch job, not something to run inside a document render: tuning alone
is 300 model fits per step, and the interaction statistics need a partial
dependence pass per predictor. Budget a couple of hours for the default sweep.

Results are cached per step under analysis/output/frontier/step_<n>/, and steps
that already have a meta.json are skipped, so the sweep can be interrupted and
resumed.

    python -m analysis.run_frontier                    # every step
    python -m analysis.run_frontier --steps 0 40 80    # selected steps
    python -m analysis.run_frontier --force            # recompute everything
    python -m analysis.run_frontier --quick            # small grids, for a smoke test
"""

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "warehouse.duckdb"
OUTPUT = ROOT / "analysis" / "output" / "frontier"


def frontier_steps(warehouse: Path) -> list[int]:
    """Step sizes from the dbt frontier summary."""
    import duckdb

    con = duckdb.connect(str(warehouse), read_only=True)
    steps = con.sql(
        "select n_dropped from mart_completeness_frontier order by n_dropped"
    ).df()
    con.close()
    return steps.n_dropped.astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=4595)
    parser.add_argument("--no-xgb", action="store_true")
    args = parser.parse_args()

    from analysis.frontier import load_frontier_step
    from analysis.pipeline import run_analysis, save_result

    steps = args.steps if args.steps is not None else frontier_steps(WAREHOUSE)

    settings = (
        dict(tune_size=8, n_repeats=10, interaction_grid=10)
        if args.quick
        else dict(tune_size=60, n_repeats=100, interaction_grid=30)
    )

    print(f"{len(steps)} steps: {steps}")
    if args.quick:
        print("quick mode: reduced grids, results are not publication quality")

    started = time.time()

    for step in steps:
        directory = OUTPUT / f"step_{step:04d}"

        if (directory / "meta.json").exists() and not args.force:
            print(f"n_dropped={step}: cached, skipping")
            continue

        wide = load_frontier_step(WAREHOUSE, step)
        print(
            f"n_dropped={step}: {wide.shape[0]} regions, "
            f"{wide.shape[1] - 1} indicators",
            flush=True,
        )

        step_started = time.time()
        result = run_analysis(
            wide, n_dropped=step, seed=args.seed,
            with_xgboost=not args.no_xgb, **settings
        )
        save_result(result, directory)

        print(
            f"    done in {(time.time() - step_started) / 60:.1f} min "
            f"-> {directory.relative_to(ROOT)}",
            flush=True,
        )

    print(f"\ntotal {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
