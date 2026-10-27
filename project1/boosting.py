"""
Gradient boosting on top of the logistic regression, NumPy only.

Idea: the logistic regression gives a first score F(x) (its logit). Then, at
each round, a small regression tree learns how to correct the remaining error
and we add a small step of it:

    F(x)  <-  F(x) + learning_rate * tree(x)
    probability = sigmoid(F(x))

For the logistic loss, the "error" of each person is described by
    g = p - y          (gradient: >0 if we predicted too high, <0 if too low)
    h = p * (1 - p)    (hessian: how confident we already are)
A leaf containing the people S predicts the correction  -sum(g) / (sum(h) + lambda)
and the best split is the one that maximizes
    G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda)
(G, H = sums of g and h in the group). This is the same rule as XGBoost.

The trees reuse the binning of trees.py (features cut into <= 32 bins).
"""
import numpy as np

from trees import N_BINS, predict_tree


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# ---------------------------------------------------------------------
# 1. Regression tree on (gradient, hessian)
# ---------------------------------------------------------------------
def best_split_gh(xb, g, h, rows, features, min_leaf, reg_lambda):
    """Best split "xb[:, f] <= b" for the gradients g and hessians h of `rows`.
    Returns (feature, bin) or None if no split has a positive gain."""
    sub = xb[np.ix_(rows, features)]
    n, m = sub.shape
    flat = (sub + N_BINS * np.arange(m)).ravel()
    size = m * N_BINS

    cnt = np.bincount(flat, minlength=size).reshape(m, N_BINS)
    G = np.bincount(flat, weights=np.repeat(g[rows], m), minlength=size).reshape(m, N_BINS)
    H = np.bincount(flat, weights=np.repeat(h[rows], m), minlength=size).reshape(m, N_BINS)

    n_left = np.cumsum(cnt, axis=1)[:, :-1]
    g_left = np.cumsum(G, axis=1)[:, :-1]
    h_left = np.cumsum(H, axis=1)[:, :-1]
    g_tot, h_tot = g[rows].sum(), h[rows].sum()
    g_right, h_right = g_tot - g_left, h_tot - h_left

    gain = (g_left ** 2 / (h_left + reg_lambda)
            + g_right ** 2 / (h_right + reg_lambda)
            - g_tot ** 2 / (h_tot + reg_lambda))
    gain[(n_left < min_leaf) | (n - n_left < min_leaf)] = -np.inf

    k = np.argmax(gain)
    f, b = divmod(k, N_BINS - 1)
    if not gain[f, b] > 1e-9:
        return None
    return features[f], b


def build_tree_gh(xb, g, h, rows, features, depth, max_depth, min_leaf, reg_lambda):
    """Grow a regression tree; a leaf predicts -sum(g) / (sum(h) + lambda)."""
    node = {"value": -g[rows].sum() / (h[rows].sum() + reg_lambda)}
    if depth >= max_depth or len(rows) < 2 * min_leaf:
        return node
    split = best_split_gh(xb, g, h, rows, features, min_leaf, reg_lambda)
    if split is None:
        return node
    f, b = split
    go_left = xb[rows, f] <= b
    node.update(feature=f, bin=b,
                left=build_tree_gh(xb, g, h, rows[go_left], features, depth + 1,
                                   max_depth, min_leaf, reg_lambda),
                right=build_tree_gh(xb, g, h, rows[~go_left], features, depth + 1,
                                    max_depth, min_leaf, reg_lambda))
    return node


# ---------------------------------------------------------------------
# 2. Boosting
# ---------------------------------------------------------------------
def train_boosting(xb, y, base_logit, n_rounds=100, learning_rate=0.1, max_depth=3,
                   min_leaf=100, reg_lambda=1.0, subsample=0.5, colsample=0.5,
                   seed=0, xb_val=None, base_val=None, callback=None):
    """
    xb         : binned training features (trees.apply_bins)
    y          : labels 0/1
    base_logit : starting score of each training row (logit of the logistic regression)
    subsample  : share of rows used by each tree (random, without replacement)
    colsample  : share of features available to each tree
    xb_val, base_val, callback : optional; after each round,
        callback(round, validation_logit) is called (to follow the validation F1)
    Returns the list of trees.
    """
    rng = np.random.default_rng(seed)
    n, d = xb.shape
    y = y.astype(float)
    F = base_logit.astype(float).copy()
    F_val = None if xb_val is None else base_val.astype(float).copy()
    trees = []
    for r in range(n_rounds):
        p = sigmoid(F)
        g, h = p - y, p * (1 - p)
        rows = rng.choice(n, int(subsample * n), replace=False)
        features = np.sort(rng.choice(d, max(1, int(colsample * d)), replace=False))
        tree = build_tree_gh(xb, g, h, rows, features, 0, max_depth, min_leaf, reg_lambda)
        trees.append(tree)
        F += learning_rate * predict_tree(tree, xb)
        if F_val is not None:
            F_val += learning_rate * predict_tree(tree, xb_val)
            if callback is not None:
                callback(r + 1, F_val)
    return trees


def predict_boosting(trees, xb, base_logit, learning_rate=0.1):
    """Probability of class 1: sigmoid(base + learning_rate * sum of the trees)."""
    F = base_logit.astype(float).copy()
    for tree in trees:
        F += learning_rate * predict_tree(tree, xb)
    return sigmoid(F)