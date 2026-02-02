# models/sarimax_model.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import optuna

from src.data_utils import prepare_train_val_test_split


# ---------------------------
# Build model with given params (no fit)
# ---------------------------
def build_sarimax(
    data: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    order: Tuple[int, int, int] = (2, 0, 2),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
    val_days: int = 28,
    test_days: int = 28,
    trend: str = "c",
):
    """
    Return an unfitted SARIMAX plus val/test exog/target and scaler.
    """
    y_train, y_val, y_test, X_train, X_val, X_test, scaler, val_start_ts, test_start_ts = prepare_train_val_test_split(
        data=data,
        target_col="load_mw",
        feature_cols=feature_cols,
        val_days=val_days,
        test_days=test_days,
        freq="h",      # match your splitter default
        tz="UTC",
        scale_X=True,
        fill_exog=True,
    )

    model = SARIMAX(
        endog=y_train,
        exog=X_train if X_train.shape[1] > 0 else None,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
        initialization="approximate_diffuse",
        missing="drop",
    )

    return (
        model,
        (X_val, y_val),
        (X_test, y_test),
        scaler,
        val_start_ts,
        test_start_ts,
    )


# ---------------------------
# Hyperparameter tuning with Optuna
# objective: "aic" or "rmse"
# ---------------------------
def tune_sarimax(
    data: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    val_days: int = 28,
    test_days: int = 28,
    objective: str = "aic",          # "aic" or "rmse"
    n_trials: int = 40,
    season_length: int = 24,
    p_range: Tuple[int, int] = (0, 3),
    d_range: Tuple[int, int] = (0, 1),
    q_range: Tuple[int, int] = (0, 3),
    P_range: Tuple[int, int] = (0, 1),
    D_fixed: int = 1,
    Q_range: Tuple[int, int] = (0, 1),
    trend: str = "c",
    refit_full: bool = True,         # fit on train+val before test forecast
) -> Dict[str, Any]:
    assert objective in {"aic", "rmse"}

    y_train, y_val, y_test, X_train, X_val, X_test, scaler, val_start_ts, test_start_ts = prepare_train_val_test_split(
        data=data,
        target_col="load_mw",
        feature_cols=feature_cols,
        val_days=val_days,
        test_days=test_days,
        freq="h",
        tz="UTC",
        scale_X=True,
        fill_exog=True,
    )

    use_val = (len(y_val) > 0) and (objective == "rmse")

    def _fit_and_score(order, seasonal_order):
        try:
            model = SARIMAX(
                endog=y_train,
                exog=X_train if X_train.shape[1] > 0 else None,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization="approximate_diffuse",
                missing="drop",
            )
            res = model.fit(disp=False)

            if not use_val:
                return res.aic, res

            steps = len(y_val)
            fc = res.get_forecast(
                steps=steps,
                exog=X_val if X_val.shape[1] > 0 else None
            )
            y_pred = fc.predicted_mean
            rmse = root_mean_squared_error(y_val, y_pred)
            return rmse, res

        except Exception:
            return np.inf, None

    def _objective(trial: optuna.Trial):
        p = trial.suggest_int("p", *p_range)
        d = trial.suggest_int("d", *d_range)
        q = trial.suggest_int("q", *q_range)
        P = trial.suggest_int("P", *P_range)
        D = D_fixed
        Q = trial.suggest_int("Q", *Q_range)
        score, _ = _fit_and_score((p, d, q), (P, D, Q, season_length))
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)

    if study.best_trial is None or not np.isfinite(study.best_value):
        return {
            "objective": objective,
            "best_score": np.inf,
            "best_order": None,
            "best_seasonal_order": None,
            "results": None,
            "scaler": scaler,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "val_start_ts": val_start_ts,
            "test_start_ts": test_start_ts,
            "study": study,
        }

    best = study.best_params
    best_order = (best["p"], best["d"], best["q"])
    best_seasonal = (best["P"], D_fixed, best["Q"], season_length)

    # Fit once on train (and compute best_score according to objective)
    best_score, best_res_train = _fit_and_score(best_order, best_seasonal)

    # Optionally refit on train+val before testing
    best_res_full = None
    test_rmse = None
    y_test_pred = None

    if len(y_test) > 0:
        if refit_full and len(y_val) > 0:
            y_full = pd.concat([y_train, y_val])
            X_full = pd.concat([X_train, X_val]) if X_train.shape[1] > 0 else None
            model_full = SARIMAX(
                endog=y_full,
                exog=X_full,
                order=best_order,
                seasonal_order=best_seasonal,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization="approximate_diffuse",
                missing="drop",
            )
            best_res_full = model_full.fit(disp=False)

            fc_test = best_res_full.get_forecast(
                steps=len(y_test),
                exog=X_test if X_train.shape[1] > 0 else None
            )
        else:
            # Forecast test directly from train-fitted model (not ideal but valid)
            fc_test = best_res_train.get_forecast(
                steps=len(y_test),
                exog=X_test if X_train.shape[1] > 0 else None
            )

        y_test_pred = fc_test.predicted_mean
        test_rmse = root_mean_squared_error(y_test, y_test_pred)

    return {
        "objective": ("rmse" if use_val else "aic"),
        "best_score": best_score,
        "best_order": best_order,
        "best_seasonal_order": best_seasonal,
        "results_train": best_res_train,
        "results_full": best_res_full,
        "test_rmse": test_rmse,
        "y_test_pred": y_test_pred,
        "scaler": scaler,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "val_start_ts": val_start_ts,
        "test_start_ts": test_start_ts,
        "study": study,
    }
