from config import SOURCES, PROCESSED
from src.io import load_raw, write_processed
from src.wellbeing import build_wellbeing
from src.reshape import prepare_long, pivot_wide
from src.merge import (
    merge_all, correct_de7_density, drop_incomplete_columns, scale2, build_appendix,
)


def main() -> None:
    wellbeing, regions = build_wellbeing()
    print(f"well-being: {wellbeing.shape}")

    frames = {}
    for name, spec in SOURCES.items():
        long = prepare_long(
            load_raw(spec["file"]),
            regions=regions,
            key_cols=spec["key_cols"],
            extra_filters=spec.get("extra_filters"),
            whitelist=spec.get("whitelist"),
            dedupe_keys=spec.get("dedupe_keys", False),
        )
        frames[name] = pivot_wide(long, spec["key_cols"])
        print(f"{name}: {frames[name].shape}")

    data = merge_all(wellbeing, frames)
    data = correct_de7_density(data).drop(columns=["Region"])
    print(f"merged: {data.shape}")

    unscaled = drop_incomplete_columns(data)
    write_processed(unscaled, "data_unscaled.parquet")
    write_processed(scale2(unscaled), "data.parquet")
    print(f"final: {unscaled.shape}")

    dimensions = build_appendix(data, PROCESSED / "appendix")
    print(dimensions.to_string(index=False))


if __name__ == "__main__":
    main()