from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../data" / "original_data" / "oecd_regional_2016"
PROCESSED = ROOT / "../data" / "processed_data"

SEED = 4595
MIN_YEAR = 2006

WELLBEING_FILE = "RWB-2016-1-EN-20161128T112241.csv"

EXCLUDED_COUNTRIES = [
    "AUS", "AUT", "BEL", "CAN", "CHL", "CHE", "CZE", "DEU", "DNK", "ESP",
    "EST", "FIN", "FRA", "GBR", "GRC", "HUN", "IRL", "ISL", "ISR", "ITA",
    "JPN", "KOR", "LUX", "MEX", "NLD", "NOR", "NZL", "POL", "PRT", "SVK",
    "SVN", "SWE", "TUR", "USA",
]

DROPPED_INDICATORS = ["GINI", "GINIB", "PVT5A", "PVT5B", "PVT6A", "PVT6B", "S80S20A"]

MEAS_KEEP = ["USD_PPP", "PER", "RATES", "GWTH_LAB_UTIL_2007", "GWTH_LAB_UTIL_2001"]

SOURCES = {
    "demographic": {"file": "REGION_DEMOGR-2016-1-EN-20161128T111326.csv", "key_cols": ["VAR", "SEX"]},
    "labour":      {"file": "REGION_LABOUR-2016-1-EN-20161128T112125.csv", "key_cols": ["VAR", "SEX"]},
    "innovation":  {"file": "REGION_INNOVATION-2016-1-EN-20161128T112220.csv", "key_cols": ["VAR"]},
    "social":      {"file": "REGION_SOCIAL-2016-1-EN-20161128T112210.csv",
                    "key_cols": ["VAR"], "dedupe_keys": True},
    "economic":    {"file": "REGION_ECONOM-2016-1-EN-20161128T111738.csv", "key_cols": ["VAR", "MEAS"],
                    "extra_filters": {"SERIES": "SNA_2008"},
                    "whitelist": ("MEAS", MEAS_KEEP)},
}
