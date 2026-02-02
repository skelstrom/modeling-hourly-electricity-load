# src/features.py
import pandas as pd
import numpy as np
import holidays
from .config import PROCESSED

US_HOLS = holidays.US()

# ─────────────────────────────────────────────────────────
# 1) Enforce a complete hourly grid (DST-safe)
# ─────────────────────────────────────────────────────────
def align_to_hourly_grid(df: pd.DataFrame, local_tz: str = "UTC") -> pd.DataFrame:
    """
    Ensure a continuous hourly time index. Convert local time to UTC.
    Keep target (load_mw) NaN where missing. Fill exogenous with ffill/bfill.
    """
    out = df.copy()

    if "ts" not in out.columns:
        raise ValueError("Expected a 'ts' timestamp column.")

    out["ts"] = pd.to_datetime(out["ts"])
    out = out.sort_values("ts")

    # Localize → UTC (handle DST gaps/duplicates)
    if out["ts"].dt.tz is None:
        ts_local = out["ts"].dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
        out.index = ts_local.tz_convert("UTC")
    else:
        out.index = out["ts"].dt.tz_convert("UTC")
    out = out.drop(columns=["ts"])

    # Full hourly UTC grid
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="UTC")
    out = out.reindex(full_idx)

    # Fill exogenous only. Leave target NaN.
    target_col = "load_mw"
    exog_cols = [c for c in out.columns if c != target_col]
    if exog_cols:
        out[exog_cols] = out[exog_cols].ffill().bfill()

    # Return ts as column
    out = out.rename_axis("ts").reset_index()
    return out

# ─────────────────────────────────────────────────────────
# 2) Impute weather columns (time-wise interpolation)
# ─────────────────────────────────────────────────────────
def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts").copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out = out.set_index("ts")
    # ensure index is tz-aware for method="time" to work consistently
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    weather_cols = [c for c in ["temp_c", "humidity", "pressure", "wind_speed", "wind_direction"] if c in out.columns]
    for col in weather_cols:
        out[col] = out[col].interpolate(method="time", limit_direction="both")
    return out.reset_index()

# ───────────────────────────────
# 3) Impute target variable
# ───────────────────────────────
def impute_load(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts").copy()
    ts = pd.to_datetime(out["ts"])
    # make a naive UTC DatetimeIndex for time interpolation
    if getattr(ts.dt, "tz", None) is None:
        idx = ts.tz_localize("UTC").tz_convert("UTC").tz_localize(None)
    else:
        idx = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    s = pd.Series(out["load_mw"].values, index=idx)

    # 1) seasonal bridge
    s = s.fillna((s.shift(24) + s.shift(-24)) / 2.0)
    # 2) same-hour median
    s = s.fillna(s.groupby(s.index.hour).transform("median"))
    # 3) time interpolation
    s = s.interpolate(method="time", limit_direction="both")

    out["load_mw"] = s.values
    return out

# ─────────────────────────────────────────────────────────
# 4) Calendar features
# ─────────────────────────────────────────────────────────
def add_calendar(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        ts = ts.dt.tz_convert(tz)

    out["hour"] = ts.dt.hour.astype("int16")
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_holiday"] = ts.dt.date.map(lambda d: int(d in US_HOLS)).astype("int8")
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    return out

# ───────────────────────────────
# 5) LAGS / ROLLING AGGREGATES
# ───────────────────────────────
def add_lags_rolls(df: pd.DataFrame, y_col: str = "load_mw") -> pd.DataFrame:
    out = df.sort_values("ts").copy()
    out = out.set_index(pd.to_datetime(out["ts"]))

    for L in [1, 24, 168]:
        out[f"{y_col}_lag{L}"] = out[y_col].shift(L)

    out["load_roll24"]   = out[y_col].rolling(window=24,  min_periods=24).mean()
    out["load_roll168"]  = out[y_col].rolling(window=168, min_periods=168).mean()
    out["temp_roll24"]   = out["temp_c"].rolling(window=24,  min_periods=24).mean()
    out["humidity_roll24"] = out["humidity"].rolling(window=24, min_periods=24).mean()

    out["load_delta24"] = out[y_col] - out[y_col].shift(24)
    out["temp_diff24"]  = out["temp_c"] - out["temp_c"].shift(24)

    out = out.reset_index(drop=True)
    return out

# ───────────────────────────────
# 6) RELATIVE / RATIO FEATURES
# ───────────────────────────────
def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-2
    safe_temp = out["temp_c"].replace(0, eps)

    out["load_to_temp_ratio"]   = out["load_mw"] / safe_temp
    out["humidity_to_pressure"] = out["humidity"] / out["pressure"].replace(0, np.nan)

    for col in ["load_to_temp_ratio", "humidity_to_pressure"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    q01, q99 = out["load_to_temp_ratio"].quantile([0.01, 0.99])
    out["load_to_temp_ratio"] = out["load_to_temp_ratio"].clip(lower=q01, upper=q99)

    out["load_to_temp_ratio"]   = out["load_to_temp_ratio"].fillna(out["load_to_temp_ratio"].median())
    out["humidity_to_pressure"] = out["humidity_to_pressure"].fillna(out["humidity_to_pressure"].median())
    return out

# ───────────────────────────────
# 7) POLYNOMIAL FEATURES
# ───────────────────────────────
def add_temp_polynomials(df: pd.DataFrame, base_c: float = 18.0) -> pd.DataFrame:
    out = df.copy()
    out["temp_c_centered"] = out["temp_c"] - base_c
    out["temp_c2"] = out["temp_c_centered"] ** 2
    return out

# ───────────────────────────────
# 8) WEATHER INTERACTIONS
# ───────────────────────────────
def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_x_humidity"] = out["temp_c"] * out["humidity"]
    radians = np.deg2rad(out["wind_direction"])
    out["wind_x"] = np.cos(radians)
    out["wind_y"] = np.sin(radians)
    return out

# ───────────────────────────────
# 9) ONE-HOT ENCODING (weather description)
# ───────────────────────────────
def one_hot_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "weather_description" in out.columns:
        out["weather_description"] = out["weather_description"].fillna("unknown")
        wx = pd.get_dummies(out["weather_description"], prefix="wx", drop_first=True)
        out = pd.concat([out.drop(columns=["weather_description"]), wx], axis=1)
    return out

# ───────────────────────────────
# 10) FINALIZATION
# ───────────────────────────────
def finalize(df: pd.DataFrame) -> str:
    must_have = [c for c in ["load_mw_lag1", "load_mw_lag24", "load_mw_lag168"] if c in df.columns]
    cleaned = df.dropna(subset=must_have).copy()  # keep imputed load_mw
    out = PROCESSED / "model_ready.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out, index=False)
    print(f"✅ Saved feature-engineered dataset → {out} | rows={len(cleaned):,}")
    return str(out)

# ───────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = align_to_hourly_grid(df, local_tz="America/Chicago")
    df = impute_weather(df)        # exogenous only
    df = impute_load(df)           # rare target gaps
    df = add_calendar(df)
    df = add_lags_rolls(df)
    df = add_ratios(df)
    df = add_temp_polynomials(df)
    df = add_weather_interactions(df)
    df = one_hot_weather(df)
    return df

if __name__ == "__main__":
    from .config import INTERIM
    path = INTERIM / "comed_hourly_joined.parquet"
    df = pd.read_parquet(path)
    df = build_features(df)
    finalize(df)
