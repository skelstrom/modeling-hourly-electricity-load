# models/gradient_boosting.py
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple, List
from models.cart import CARTRegressor

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

class GradientBoostingRegressor:
    """
    Minimal Gradient Boosting (squared-error) with regularization:
      - shrinkage (learning_rate)
      - subsampling (rows) and feature subsampling (columns)
      - early stopping on a validation set

    Parameters
    ----------
    n_estimators : int
    learning_rate : float
    max_depth, min_samples_split, min_samples_leaf : int
        Passed to CARTRegressor (weak learner).
    subsample : float in (0,1]
        Fraction of rows used to fit each tree (stochastic gradient boosting).
    colsample : float in (0,1]
        Fraction of features randomly selected for each tree.
    early_stopping_rounds : int | None
        If set and eval_set is provided, stop if no improvement after this many rounds.
    eval_metric : Callable[[ndarray, ndarray], float] | None
        If None, uses RMSE (lower is better).
    random_state : int
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        colsample: float = 1.0,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric or _rmse
        self.random_state = random_state

        # learned state
        self.init_value_: float | None = None
        self.trees_: List[CARTRegressor] = []
        self.feature_indices_: List[np.ndarray] = []  # per-tree column subset
        self.train_history_: List[float] = []
        self.val_history_: List[float] = []
        self.best_iteration_: Optional[int] = None

    # ------------------- fit -------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        if not (0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1].")
        if not (0 < self.colsample <= 1.0):
            raise ValueError("colsample must be in (0, 1].")

        # validation set (optional)
        X_val = y_val = None
        use_es = self.early_stopping_rounds is not None and eval_set is not None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float).ravel()

        # init
        self.init_value_ = y.mean()
        y_pred = np.full(n, self.init_value_, dtype=float)
        self.trees_.clear()
        self.feature_indices_.clear()
        self.train_history_.clear()
        self.val_history_.clear()
        self.best_iteration_ = None

        best_score = np.inf
        rounds_since_improve = 0

        for m in range(self.n_estimators):
            # --- residuals for squared error ---
            residual = y - y_pred

            # --- row subsampling ---
            if self.subsample < 1.0:
                msize = max(1, int(self.subsample * n))
                rows = rng.choice(n, size=msize, replace=False)
            else:
                rows = slice(None)

            # --- feature subsampling ---
            if self.colsample < 1.0:
                fsize = max(1, int(self.colsample * d))
                cols = rng.choice(d, size=fsize, replace=False)
            else:
                cols = np.arange(d)

            X_fit = X[rows][:, cols]
            r_fit = residual[rows] if not isinstance(rows, slice) else residual

            # --- train a weak learner on residuals ---
            tree = CARTRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.integers(0, 1_000_000),
            )
            tree.fit(X_fit, r_fit)

            # --- update predictions on full X using the same feature subset ---
            update_full = tree.predict(X[:, cols])
            y_pred += self.learning_rate * update_full

            # store
            self.trees_.append(tree)
            self.feature_indices_.append(np.array(cols, dtype=int))

            # --- training metric ---
            train_score = self.eval_metric(y, y_pred)
            self.train_history_.append(train_score)

            # --- early stopping check (if any) ---
            if use_es:
                # build y_pred_val incrementally (efficiently) by applying only this tree
                if m == 0:
                    y_pred_val = np.full(X_val.shape[0], self.init_value_, dtype=float)
                else:
                    # reuse from previous iteration
                    y_pred_val = self._last_val_pred_
                y_pred_val = y_pred_val + self.learning_rate * tree.predict(X_val[:, cols])
                self._last_val_pred_ = y_pred_val

                val_score = self.eval_metric(y_val, y_pred_val)
                self.val_history_.append(val_score)

                improved = val_score + 1e-12 < best_score
                if improved:
                    best_score = val_score
                    self.best_iteration_ = m
                    rounds_since_improve = 0
                else:
                    rounds_since_improve += 1

                if verbose:
                    print(f"[{m+1:03d}] train={train_score:.5f}  val={val_score:.5f} "
                          f"{'(*)' if improved else ''}")

                if rounds_since_improve >= self.early_stopping_rounds:
                    # prune trees to best iteration
                    keep = (self.best_iteration_ or 0) + 1
                    self.trees_ = self.trees_[:keep]
                    self.feature_indices_ = self.feature_indices_[:keep]
                    # trim histories to the same length
                    self.train_history_ = self.train_history_[:keep]
                    self.val_history_ = self.val_history_[:keep]
                    return self

            elif verbose:
                print(f"[{m+1:03d}] train={train_score:.5f}")

        # if no early stopping used, set best_iteration_ to last
        if self.best_iteration_ is None:
            self.best_iteration_ = len(self.trees_) - 1
        return self

    # ------------------- predict -------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y_pred = np.full(X.shape[0], self.init_value_, dtype=float)
        for tree, cols in zip(self.trees_, self.feature_indices_):
            y_pred += self.learning_rate * tree.predict(X[:, cols])
        return y_pred
