# src/sql_build.py  
from pathlib import Path
import duckdb
from .config import RAW, INTERIM

def build_joined_hourly(use_left_join: bool = False) -> Path:
    """
    Build the joined hourly dataset (load + weather) and save to:
      data/interim/comed_hourly_joined.parquet

    Handles mixed timestamp formats like '10/13/12 0:00' by normalizing single-digit hours
    and trying multiple strptime patterns. Executes everything on a single DuckDB connection.
    """
    out = INTERIM / "comed_hourly_joined.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")

    # Register CSVs on THIS connection
    con.execute(f"CREATE OR REPLACE VIEW load AS SELECT * FROM read_csv_auto('{(RAW/'COMED_hourly.csv').as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW tmp  AS SELECT * FROM read_csv_auto('{(RAW/'temperature.csv').as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW hum  AS SELECT * FROM read_csv_auto('{(RAW/'humidity.csv').as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW prs  AS SELECT * FROM read_csv_auto('{(RAW/'pressure.csv').as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW wspd AS SELECT * FROM read_csv_auto('{(RAW/'wind_speed.csv').as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW wdir AS SELECT * FROM read_csv_auto('{(RAW/'wind_direction.csv').as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW wdes AS SELECT * FROM read_csv_auto('{(RAW/'weather_description.csv').as_posix()}')")

    # Timestamp parser (normalize single-digit hour " 0:" -> " 0 0:" via zero-pad)
    PARSE = """
      COALESCE(
        TRY_STRPTIME(REGEXP_REPLACE(dt_raw, ' (\\d):', ' 0\\1:'), '%m/%d/%y %H:%M'),
        TRY_STRPTIME(REGEXP_REPLACE(dt_raw, ' (\\d):', ' 0\\1:'), '%m/%d/%Y %H:%M'),
        TRY_STRPTIME(dt_raw, '%Y-%m-%d %H:%M:%S'),
        TRY_STRPTIME(dt_raw, '%Y-%m-%d %H:%M')
      )
    """

    join = "LEFT JOIN" if use_left_join else "INNER JOIN"

    query = f"""
    WITH
    base AS (
      SELECT
        {PARSE.replace('dt_raw', "CAST(Datetime AS VARCHAR)")} AS ts,
        CAST(COMED_MW AS DOUBLE) AS load_mw
      FROM load
    ),
    t AS (
      SELECT
        {PARSE.replace('dt_raw', "CAST(datetime AS VARCHAR)")} AS ts,
        (Chicago - 273.15) AS temp_c
      FROM tmp
    ),
    h AS (
      SELECT {PARSE.replace('dt_raw', "CAST(datetime AS VARCHAR)")} AS ts, Chicago AS humidity FROM hum
    ),
    p AS (
      SELECT {PARSE.replace('dt_raw', "CAST(datetime AS VARCHAR)")} AS ts, Chicago AS pressure FROM prs
    ),
    s AS (
      SELECT {PARSE.replace('dt_raw', "CAST(datetime AS VARCHAR)")} AS ts, Chicago AS wind_speed FROM wspd
    ),
    d AS (
      SELECT {PARSE.replace('dt_raw', "CAST(datetime AS VARCHAR)")} AS ts, Chicago AS wind_direction FROM wdir
    ),
    wd AS (
      SELECT {PARSE.replace('dt_raw', "CAST(datetime AS VARCHAR)")} AS ts, Chicago AS weather_description FROM wdes
    )
    SELECT
      b.ts,
      b.load_mw,
      ROUND(t.temp_c, 2) AS temp_c,
      h.humidity,
      p.pressure,
      s.wind_speed,
      d.wind_direction,
      wd.weather_description
    FROM base b
      {join} t  USING (ts)
      {join} h  USING (ts)
      {join} p  USING (ts)
      {join} s  USING (ts)
      {join} d  USING (ts)
      {join} wd USING (ts)
    WHERE b.ts IS NOT NULL
    ORDER BY b.ts
    """

    # IMPORTANT: execute on the SAME connection
    df = con.execute(query).df()
    if df.empty:
        con.close()
        raise RuntimeError("Joined dataframe is empty; check timestamp parsing and input CSV columns.")

    df.to_parquet(out, index=False)
    con.close()
    print(f"✅ Joined dataset written to: {out}")
    print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
    return out
