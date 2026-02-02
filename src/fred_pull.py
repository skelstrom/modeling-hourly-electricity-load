# src/fred_pull.py
from pathlib import Path
import os
import pandas as pd
from fredapi import Fred

# ─── Directories ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"  
RAW.mkdir(parents=True, exist_ok=True)

# ─── FRED API Setup ──────────────────────────────────────
fred = Fred(api_key=os.getenv("FRED_API_KEY"))
if fred is None:
    raise ValueError("FRED_API_KEY environment variable not found. Please set it before running.")

# ─── Series Config ───────────────────────────────────────
SERIES = {
    "INDPRO": "industrial_production",
    "DCOILWTICO": "wti_oil_price",
    "UNRATE": "us_unemployment_rate",
}

# ─── Main Function ───────────────────────────────────────
def pull_fred(start="2012-10-01", end="2017-11-30") -> Path:
    """Pull selected FRED economic indicators and save as parquet."""
    frames = []
    for sid, name in SERIES.items():
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        df = s.to_frame(name=name)
        df.index.name = "date"
        frames.append(df)

    out = RAW / "fred_monthly.parquet"
    pd.concat(frames, axis=1).to_parquet(out)
    print(f"✅ Wrote {out}")
    return out

# ─── CLI Entry ───────────────────────────────────────────
if __name__ == "__main__":
    pull_fred()