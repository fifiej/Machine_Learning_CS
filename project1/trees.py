"""
Decision tree and random forest for binary classification, NumPy only.

Main ideas:
  * Features are first cut into at most N_BINS bins (quantiles computed on the
    training data). A split is then "bin <= b", so for one feature we only
    test ~32 thresholds instead of every distinct value -> fast.
  * The quality of a split is the Gini impurity of the two children,
    weighted by their size. Gini of a group with a proportion p of positives:
        gini(p) = 1 - p^2 - (1 - p)^2 = 2 p (1 - p)
  * A leaf predicts the proportion of positives it contains (a probability),
    so we can choose the decision threshold that maximizes F1, as for the
    logistic regression.
  * Random forest = many trees, each trained on a bootstrap sample and
    choosing each split among a random subset of features; the predicted
    probability is the average over the trees.
"""
import numpy as np

N_BINS = 32


# ---------------------------------------------------------------------
# 1. Binning (learned on x_train, applied to any data)
# ---------------------------------------------------------------------
def fit_bins(x, n_bins=N_BINS):
    """Fill NaNs with the column mean and compute the bin edges of each column.
    Returns (mean, edges) where edges[j] are the cut points of column j."""
    mean = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), mean, x)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(x[:, j], qs)) for j in range(x.shape[1])]
    return mean, edges


def apply_bins(x, mean, edges):
    """Replace each value by the index of its bin (0, 1, 2, ...).
    value <= edges[j][b]  <=>  bin <= b"""
    x = np.where(np.isnan(x), mean, x)
    xb = np.empty(x.shape, dtype=np.int64)
    for j, e in enumerate(edges):
        xb[:, j] = np.searchsorted(e, x[:, j], side="left")
    return xb


# ---------------------------------------------------------------------
# 2. Gini impurity and best split
# ---------------------------------------------------------------------
def gini(y):
    """Gini impurity of a group of 0/1 labels (0 = pure, 0.5 = half-half)."""
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1 - p ** 2 - (1 - p) ** 2


def best_split(xb, y, rows, features, n_bins, min_leaf):
    """Find the split "xb[:, f] <= b" with the lowest weighted Gini impurity,
    among the given features, for the samples `rows`.
    Returns (feature, bin) or None if no split improves the impurity."""
    sub = xb[np.ix_(rows, features)]                     # n x m bin indices
    n, m = sub.shape
    y_rows = y[rows]

    # count, for every (feature, bin), the number of samples and of positives
    flat = (sub + n_bins * np.arange(m)).ravel()
    total = np.bincount(flat, minlength=m * n_bins).reshape(m, n_bins)
    pos = np.bincount(flat, weights=np.repeat(y_rows, m),
                      minlength=m * n_bins).reshape(m, n_bins)

    # left child = bins 0..b, right child = the rest (for every possible b)
    n_left = np.cumsum(total, axis=1)[:, :-1]
    p_left = np.cumsum(pos, axis=1)[:, :-1]
    n_right = n - n_left
    p_right = y_rows.sum() - p_left

    # weighted Gini = (n_L * gini_L + n_R * gini_R) / n ; gini = 2 p (1 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        impurity = (p_left * (n_left - p_left) / n_left
                    + p_right * (n_right - p_right) / n_right)
    impurity[(n_left < min_leaf) | (n_right < min_leaf)] = np.inf

    k = np.argmin(impurity)
    f, b = divmod(k, n_bins - 1)
    parent = y_rows.sum() * (n - y_rows.sum()) / n
    if not np.isfinite(impurity[f, b]) or impurity[f, b] >= parent - 1e-12:
        return None
    return features[f], b


# ---------------------------------------------------------------------
# 3. Decision tree
# ---------------------------------------------------------------------
def build_tree(xb, y, rows, depth, max_depth, min_leaf, max_features, rng):
    """Grow a tree recursively. A node is a dict:
       leaf:          {"value": p}
       internal node: {"value": p, "feature": f, "bin": b, "left": ..., "right": ...}"""
    p = y[rows].mean()
    node = {"value": p}
    if depth >= max_depth or len(rows) < 2 * min_leaf or p == 0 or p == 1:
        return node

    n_features = xb.shape[1]
    if max_features is None or max_features >= n_features:
        features = np.arange(n_features)
    else:
        features = rng.choice(n_features, max_features, replace=False)

    split = best_split(xb, y, rows, features, N_BINS, min_leaf)
    if split is None:
        return node

    f, b = split
    go_left = xb[rows, f] <= b
    node.update(feature=f, bin=b,
                left=build_tree(xb, y, rows[go_left], depth + 1,
                                max_depth, min_leaf, max_features, rng),
                right=build_tree(xb, y, rows[~go_left], depth + 1,
                                 max_depth, min_leaf, max_features, rng))
    return node


def train_tree(xb, y, max_depth=8, min_leaf=50, max_features=None, rows=None, seed=0):
    """Train one decision tree on the binned data xb and labels y (0/1)."""
    rng = np.random.default_rng(seed)
    if rows is None:
        rows = np.arange(len(y))
    return build_tree(xb, y.astype(float), rows, 0, max_depth, min_leaf, max_features, rng)


def predict_tree(node, xb):
    """Predicted probability of class 1 for every row of xb."""
    out = np.empty(len(xb))
    stack = [(node, np.arange(len(xb)))]
    while stack:
        nd, rows = stack.pop()
        if "feature" not in nd:          # leaf
            out[rows] = nd["value"]
            continue
        go_left = xb[rows, nd["feature"]] <= nd["bin"]
        stack.append((nd["left"], rows[go_left]))
        stack.append((nd["right"], rows[~go_left]))
    return out


def tree_depth(node):
    if "feature" not in node:
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


# ---------------------------------------------------------------------
# 4. Random forest
# ---------------------------------------------------------------------
def train_forest(xb, y, n_trees=50, max_depth=10, min_leaf=50, max_features="sqrt",
                 sample_size=None, seed=0, verbose=True):
    """Train n_trees trees, each on a bootstrap sample (drawn with replacement)
    and choosing each split among max_features random features."""
    rng = np.random.default_rng(seed)
    n, d = xb.shape
    if max_features == "sqrt":
        max_features = max(1, int(np.sqrt(d)))
    if sample_size is None:
        sample_size = n
    trees = []
    for t in range(n_trees):
        rows = rng.integers(0, n, sample_size)          # bootstrap sample
        trees.append(train_tree(xb, y, max_depth, min_leaf, max_features,
                                rows=rows, seed=int(rng.integers(1 << 31))))
        if verbose and (t + 1) % 10 == 0:
            print(f"  {t + 1}/{n_trees} trees")
    return trees


def predict_forest(trees, xb):
    """Average of the probabilities predicted by the trees."""
    return np.mean([predict_tree(tree, xb) for tree in trees], axis=0)
