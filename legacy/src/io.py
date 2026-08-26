import pandas as pd
from config import RAW, PROCESSED


def load_raw(filename: str) -> pd.DataFrame:
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available: {sorted(p.name for p in RAW.glob('*.csv'))}"
        )
    return pd.read_csv(path, low_memory=False)


def write_processed(df: pd.DataFrame, filename: str, subdir: str | None = None) -> None:
    out_dir = PROCESSED / subdir if subdir else PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / filename, index=False)