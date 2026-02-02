# src/make_dataset.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from .config import RAW, INTERIM, PROCESSED
from .sql_build import build_joined_hourly
from .fred_pull import pull_fred
from .enrich_econ import join_econ
from .features import build_features, finalize


def run_pipeline(preview_rows: int = 10) -> Path:
    # 0) Load .env so new users can just set FRED_API_KEY once
    load_dotenv()
    if not os.getenv("FRED_API_KEY"):
        msg = (
            "FRED_API_KEY not found.\n"
            "Create a file named `.env` in the project root with this line:\n"
            'FRED_API_KEY="YOUR_FRED_KEY_HERE"\n'
            "Then re-run: python -m src.make_dataset"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    # 1) Hourly load + weather -> interim
    hourly_path = build_joined_hourly()
    print(f"✅ Hourly written → {hourly_path}")

    # 2) FRED econ -> raw
    fred_path = pull_fred()
    print(f"✅ FRED written → {fred_path}")

    # 3) Enrich hourly with econ -> processed (intermediate)
    enriched_path = join_econ(hourly_path=hourly_path, fred_path=fred_path)
    print(f"✅ Enriched hourly+econ → {enriched_path}")

    # 4) Feature engineering -> final processed
    df = pd.read_parquet(enriched_path)
    df_feat = build_features(df)
    final_path = Path(finalize(df_feat))

    # 5) Quick preview + missingness
    print("\n──────── Preview ────────")
    print(df_feat.head(preview_rows))
    print("\n──────── Missingness (top 10) ────────")
    print(df_feat.isna().mean().sort_values(ascending=False).head(10))

    print(f"\n✅ Final dataset → {final_path}")
    return final_path


if __name__ == "__main__":
    run_pipeline(preview_rows=10)
