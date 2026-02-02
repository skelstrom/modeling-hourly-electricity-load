# src/data_utils.py
from __future__ import annotations
import numpy as np
import torch
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def make_windows(X, y, enc_len, dec_len, dec_mask_cols=None):
    # X: (N, n_features)
    # y: (N,) or (N, y_dim)
    import numpy as np
    import torch

    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim == 1:
        y = y[:, None]  # (N, 1)

    n = len(X) - (enc_len + dec_len) + 1
    Xenc = np.zeros((n, enc_len, X.shape[1]), dtype=np.float32)
    Xdec = np.zeros((n, dec_len, X.shape[1]), dtype=np.float32)
    Y    = np.zeros((n, dec_len, y.shape[1]), dtype=np.float32)
    y_enc_last = np.zeros((n, y.shape[1]), dtype=np.float32)

    for i in range(n):
        s = i
        e = i + enc_len
        d = e + dec_len

        Xenc[i] = X[s:e]
        Xdec[i] = X[e:d]
        Y[i]    = y[e:d]
        y_enc_last[i] = y[e-1]   # last observed y at end of encoder

    if dec_mask_cols is not None and len(dec_mask_cols) > 0:
        Xdec[:, :, dec_mask_cols] = 0.0

    return (
        torch.from_numpy(Xenc),
        torch.from_numpy(Xdec),
        torch.from_numpy(Y),
        torch.from_numpy(y_enc_last),
    )



def prepare_train_val_test_split(
    data: pd.DataFrame,
    target_col: str = "load_mw",
    feature_cols: Optional[List[str]] = None,
    val_days: int = 28,
    test_days: int = 28,
    freq: str = "H",
    tz: str = "UTC",
    scale_X: bool = True,
    fill_exog: bool = True,
) -> Tuple[
    pd.Series, pd.Series, pd.Series,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    Optional[StandardScaler],
    pd.Timestamp, pd.Timestamp
]:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("`data` must have a DatetimeIndex.")

    df = data.copy().sort_index()

    # Ensure tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    y = df[target_col].copy()
    X = df[feature_cols].copy() if len(feature_cols) else pd.DataFrame(index=df.index)

    # Align to regular grid
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=tz)
    y = y.reindex(full_idx)
    X = X.reindex(full_idx)

    # Drop rows where target is missing (critical for SARIMAX/XGB/TFT)
    mask = y.notna()
    y = y.loc[mask]
    X = X.loc[y.index]

    # Fill exogenous gaps (safe-ish); do NOT fill y here
    if fill_exog and X.shape[1] > 0:
        X = X.ffill().bfill()

    # Determine steps per day from freq
    offset = pd.tseries.frequencies.to_offset(freq)
    day = pd.Timedelta(days=1)

    # Only supports frequencies that evenly divide a day
    steps_per_day = int(day / pd.Timedelta(offset.delta))
    val_steps = val_days * steps_per_day
    test_steps = test_days * steps_per_day

    n = len(y)
    if n <= (val_steps + test_steps + 1):
        raise ValueError(
            f"Not enough data after alignment: n={n}, need > val_steps+test_steps={val_steps+test_steps}."
        )

    # Split by position (no off-by-one ambiguity)
    test_start_i = n - test_steps
    val_start_i  = n - (test_steps + val_steps)

    y_train = y.iloc[:val_start_i]
    y_val   = y.iloc[val_start_i:test_start_i]
    y_test  = y.iloc[test_start_i:]

    X_train = X.loc[y_train.index]
    X_val   = X.loc[y_val.index]
    X_test  = X.loc[y_test.index]

    scaler = None
    if scale_X and X_train.shape[1] > 0:
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_val   = pd.DataFrame(scaler.transform(X_val),   index=X_val.index,   columns=X_val.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),  index=X_test.index,  columns=X_test.columns)

    # Split timestamps (first timestamp of each split)
    val_start_ts = y_val.index.min()
    test_start_ts = y_test.index.min()

    return y_train, y_val, y_test, X_train, X_val, X_test, scaler, val_start_ts, test_start_ts


def save_sarimax_bundle(tuned: dict, model_dir: Path, name: str):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    res = tuned.get("results_full") or tuned.get("results_train")
    if res is None:
        raise ValueError("No SARIMAX results object to save.")

    # Save statsmodels results
    res_path = model_dir / f"{name}.pkl"
    res.save(res_path)

    # Save metadata needed to reproduce/validate
    meta = {
        "objective": tuned.get("objective"),
        "best_score": float(tuned.get("best_score")) if tuned.get("best_score") is not None else None,
        "best_order": tuned.get("best_order"),
        "best_seasonal_order": tuned.get("best_seasonal_order"),
        "feature_cols": list(tuned.get("X_val").columns) if tuned.get("X_val") is not None else None,
        "val_start_ts": str(tuned.get("val_start_ts")) if tuned.get("val_start_ts") is not None else None,
        "test_start_ts": str(tuned.get("test_start_ts")) if tuned.get("test_start_ts") is not None else None,
        "refit_full": tuned.get("results_full") is not None,
    }
    meta_path = model_dir / f"{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return res_path, meta_path

def load_sarimax_bundle(model_dir: Path, name: str):
    model_dir = Path(model_dir)
    res_path = model_dir / f"{name}.pkl"
    meta_path = model_dir / f"{name}.json"

    res = SARIMAXResults.load(res_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return res, meta


def save_tft_bundle(
    model: torch.nn.Module,
    model_dir: Path,
    name: str,
    config: dict,
    metrics: dict,
    overwrite: bool = False,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / f"{name}.pt"
    meta_path = model_dir / f"{name}.json"

    if ckpt_path.exists() and (not overwrite):
        raise FileExistsError(f"{ckpt_path} exists. Set overwrite=True or change name.")

    # checkpoint (CPU tensors)
    ckpt = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "metrics": metrics,
    }
    torch.save(ckpt, ckpt_path)

    meta = {"config": config, "metrics": metrics}
    meta_path.write_text(json.dumps(meta, indent=2))

    return ckpt_path, meta_path

def load_tft_bundle(
    model: torch.nn.Module,
    model_dir: Path,
    name: str,
    device: torch.device,
):
    model_dir = Path(model_dir)
    ckpt_path = model_dir / f"{name}.pt"
    meta_path = model_dir / f"{name}.json"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    meta = ckpt.get("metrics", {})
    if meta_path.exists():
        meta_json = json.loads(meta_path.read_text())
    else:
        meta_json = {}

    return model, ckpt_path, meta_json


