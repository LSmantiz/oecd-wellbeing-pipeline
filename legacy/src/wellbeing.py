# src/wellbeing.py
import pandas as pd
from config import RAW, WELLBEING_FILE, EXCLUDED_COUNTRIES, DROPPED_INDICATORS
from src.reshape import keep_latest


def build_wellbeing() -> tuple[pd.DataFrame, list[str]]:
    """Build the wide well-being table and the list of usable regions."""
    wb = pd.read_csv(RAW / WELLBEING_FILE)
    wb = wb[~wb["REG_ID"].isin(EXCLUDED_COUNTRIES)]
    wb = keep_latest(wb, ["REG_ID", "Regions", "IND", "MEAS"])

    wide = (
        wb.loc[wb["MEAS"] == "VALUE", ["REG_ID", "Regions", "IND", "Value"]]
        .pivot(index=["REG_ID", "Regions"], columns="IND", values="Value")
        .reset_index()
        .rename_axis(columns=None)
        .dropna(subset=["SUBJ_LIFE_SAT"])
        .drop(columns=DROPPED_INDICATORS)
    )
    wide = wide.rename(columns={"Regions": "Region"})
    indicator_cols = [c for c in wide.columns if c not in ("REG_ID", "Region")]
    wide = wide[wide[indicator_cols].notna().all(axis=1)].reset_index(drop=True)

    return wide, sorted(wide["REG_ID"].unique())