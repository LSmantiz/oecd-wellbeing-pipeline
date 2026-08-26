from pathlib import Path
import duckdb

SRC = Path("data/original_data/oecd_regional_2016")
DST = Path("tests/fixtures/oecd_regional_2016")
DST.mkdir(parents=True, exist_ok=True)

# a handful of regions plus their country rows, so the TL filter is exercised
KEEP = ("'DE7','DE1','DE2','AT1','AT2','DEU','AUT'")

con = duckdb.connect()
for csv in SRC.glob("*.csv"):
    con.sql(f"""
        copy (
            select * from read_csv_auto('{csv}', sample_size = -1, all_varchar = true)
            where "REG_ID" in ({KEEP})
        ) to '{DST / csv.name}' (header, delimiter ',')
    """)
    print(csv.name, (DST / csv.name).stat().st_size // 1024, "KB")