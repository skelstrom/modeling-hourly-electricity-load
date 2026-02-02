# src/enrich_econ.py
from pathlib import Path
import pandas as pd

from .config import INTERIM, PROCESSED, EXTERNAL

def join_econ(
    hourly_path: Path | None = None,
    fred_path: Path | None = None
) -> Path:
    """
    Join monthly/daily FRED macro features to hourly load+weather by ffill to hourly.
    """
    # Resolve default paths if not provided
    hourly_path = hourly_path or (INTERIM / "comed_hourly_joined.parquet")
    fred_path = fred_path or (EXTERNAL / "fred_monthly.parquet")

    # --- Load hourly dataset
    h = pd.read_parquet(hourly_path)
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    h = h.sort_values("ts").drop_duplicates(subset=["ts"])

    # --- Load FRED dataset (index = date)
    m = pd.read_parquet(fred_path)
    # ensure index is datetime with UTC tz
    if not isinstance(m.index, pd.DatetimeIndex):
        m.index = pd.to_datetime(m.index)
    if m.index.tz is None:
        m.index = m.index.tz_localize("UTC")
    else:
        m.index = m.index.tz_convert("UTC")

    # --- Reindex monthly to a continuous hourly timeline, forward-fill
    idx = pd.date_range(h["ts"].min(), h["ts"].max(), freq="h", tz="UTC")
    m_hourly = (
        m.reindex(idx)
         .ffill()
         .reset_index()
         .rename(columns={"index": "ts"})
    )

    # --- Join on timestamp
    df = h.merge(m_hourly, on="ts", how="left")

    # --- Write processed
    out = PROCESSED / "model_ready_with_econ.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out

if __name__ == "__main__":
    out = join_econ()
    print("✅ Wrote", out)
