# models/cart.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class _Node:
    is_leaf: bool
    value: float = 0.0
    feature: int | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None
    nan_left: bool | None = None

class CARTRegressor:
    """
    Minimal CART for regression (MSE impurity).
    - Numeric features only.
    - Handles NaNs by sending them to the child with more samples (simple, fast).
    """
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: int | None = None,
        random_state: int | None = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._nodes: list[_Node] = []
        self.n_features_: int = 0

    # ------------------------- public API -------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARTRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        feat_idx_all = np.arange(self.n_features_)
        if self.max_features is None or self.max_features > self.n_features_:
            feat_subset = feat_idx_all
        else:
            feat_subset = rng.choice(feat_idx_all, size=self.max_features, replace=False)

        self._nodes = []
        # build returns node index
        self._build(X, y, depth=0, feat_candidates=feat_subset)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._nodes:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        preds = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            preds[i] = self._predict_row(X[i])
        return preds

    # ------------------------- core tree logic -------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, feat_candidates: np.ndarray) -> int:
        # stopping conditions
        n, _ = X.shape
        leaf_value = float(y.mean()) if n > 0 else 0.0
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return self._add_leaf(leaf_value)

        # find best split
        best_feat, best_thr, best_gain = self._best_split(X, y, feat_candidates)
        if best_feat is None:
            return self._add_leaf(leaf_value)

        # partition
        col = X[:, best_feat]
        left_mask = col <= best_thr
        right_mask = ~left_mask

        # route NaNs: send to side with more samples
        nan_mask = np.isnan(col)
        nan_left = None
        if nan_mask.any():
            nan_left = left_mask.sum() >= right_mask.sum()
            if nan_left:
                left_mask |= nan_mask
            else:
                right_mask |= nan_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self._add_leaf(leaf_value)

        node_idx = len(self._nodes)
        self._nodes.append(_Node(
            is_leaf=False, value=leaf_value, feature=best_feat,
            threshold=best_thr, nan_left=nan_left
        ))

        left_idx = self._build(X_left, y_left, depth + 1, feat_candidates)
        right_idx = self._build(X_right, y_right, depth + 1, feat_candidates)

        self._nodes[node_idx].left = left_idx
        self._nodes[node_idx].right = right_idx
        return node_idx

    def _add_leaf(self, value: float) -> int:
        idx = len(self._nodes)
        self._nodes.append(_Node(True, value=value))
        return idx

    # ------------------------- split search (MSE) -------------------------

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        # variance proxy times n: sum((y - mean)^2) / n minimized is equivalent to SSE
        mu = y.mean()
        return float(((y - mu) ** 2).mean())

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_candidates: np.ndarray):
        best_feat = None
        best_thr = None
        best_gain = 0.0

        for j in feat_candidates:
            xj = X[:, j]

            # drop NaNs for threshold search
            mask = ~np.isnan(xj)
            xs = xj[mask]
            ys = y[mask]
            n_eff = len(ys)

            # enough points to make two leaves and roughly worth evaluating
            if n_eff < max(2 * self.min_samples_leaf, self.min_samples_split // 2):
                continue

            # sort by feature
            order = np.argsort(xs, kind="mergesort")
            xs = xs[order]
            ys = ys[order]

            # no variation => no split
            if xs[0] == xs[-1]:
                continue

            parent_imp_feat = self._mse(ys)

            # prefix sums for fast left/right stats
            csum = np.cumsum(ys)
            csum2 = np.cumsum(ys * ys)

            for k in range(self.min_samples_leaf, n_eff - self.min_samples_leaf + 1):
                # skip identical-value boundary
                if k < n_eff and xs[k] == xs[k - 1]:
                    continue

                nL = k
                nR = n_eff - k

                sumL = csum[k - 1]
                sum2L = csum2[k - 1]
                sumR = csum[-1] - sumL
                sum2R = csum2[-1] - sum2L

                mseL = (sum2L / nL) - (sumL / nL) ** 2
                mseR = (sum2R / nR) - (sumR / nR) ** 2

                # impurity decrease using feature-local stats
                gain = parent_imp_feat - (nL / n_eff) * mseL - (nR / n_eff) * mseR
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thr = 0.5 * (xs[k - 1] + xs[k])  # midpoint

        return best_feat, best_thr, best_gain


    # ------------------------- inference -------------------------

    def _predict_row(self, x: np.ndarray) -> float:
        node_idx = 0
        while True:
            node = self._nodes[node_idx]
            if node.is_leaf:
                return node.value
            feat, thr = node.feature, node.threshold
            xv = x[feat]
            # NaN routing: go to bigger child recorded at build time via threshold rule
            if np.isnan(xv):
                # choose child based on which side threshold would send majority during fit
                # approximate: send to right if threshold is very small and value is NaN
                # simpler: send to left (consistent with fit’s bias)
                node_idx = node.left if (node.nan_left is None or node.nan_left) else node.right  
            else:
                node_idx = node.left if xv <= thr else node.right
