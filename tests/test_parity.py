from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_mart_reproduces_published_dataset():
    con = duckdb.connect(str(ROOT / "warehouse.duckdb"))
    new = (con.sql("select reg_id, indicator, value from mart_analysis_long").df()
             .pivot(index="reg_id", columns="indicator", values="value"))
    old = pd.read_parquet(ROOT / "data/processed_data/data_unscaled.parquet").set_index("REG_ID")

    assert set(new.index) == set(old.index)
    assert set(new.columns) == set(old.columns)
    diff = (new.loc[old.index, old.columns] - old).abs()
    assert diff.max().max() < 1e-9