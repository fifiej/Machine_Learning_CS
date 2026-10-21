"""
Shared preprocessing for the BRFSS project.
Everyone in the group imports from this file, so that we all use
exactly the same cleaning, split and evaluation.
"""

import numpy as np

SEED = 0  # same split for everyone -> comparable F1 scores


# ---------------------------------------------------------------------
# 1. Column groups (computed on x_train only, then reused for x_test)
# ---------------------------------------------------------------------
CODE_FAMILIES = {
    9: [7, 8, 9],
    99: [77, 88, 99],
    999: [777, 888, 999],
    9999: [7777, 8888, 9999],
}


def compute_groups(x, names):
    """Sort columns into groups that share the same cleaning rule."""
    groups = {
        "mostly missing (>90% NaN)": [],
        "constant": [],
        "binary with codes (1/2 + 7/9)": [],
        "binary without codes": [],
        "small scale with codes": [],
        "numeric with codes": [],
        "no codes detected": [],
    }
    for j, name in enumerate(names):
        col = x[:, j]
        values = col[~np.isnan(col)]
        nan_ratio = np.isnan(col).mean()
        if len(values) == 0:
            groups["mostly missing (>90% NaN)"].append(name)
            continue
        uniques = np.unique(values)
        vmax = uniques.max()

        codes, real_max = [], vmax
        if vmax in CODE_FAMILIES:
            family = CODE_FAMILIES[vmax]
            real_values = uniques[~np.isin(uniques, family)]
            if len(real_values) > 0 and real_values.max() < min(family):
                codes = [c for c in family if c in uniques]
                real_max = real_values.max()

        if nan_ratio > 0.9:
            groups["mostly missing (>90% NaN)"].append(name)
        elif len(uniques) <= 1:
            groups["constant"].append(name)
        elif real_max <= 2 and codes:
            groups["binary with codes (1/2 + 7/9)"].append(name)
        elif real_max <= 2:
            groups["binary without codes"].append(name)
        elif real_max <= 10 and codes:
            groups["small scale with codes"].append(name)
        elif codes:
            groups["numeric with codes"].append(name)
        else:
            groups["no codes detected"].append(name)
    return groups


# ---------------------------------------------------------------------
# 2. Cleaning (rules based on column names -> same for train and test)
# ---------------------------------------------------------------------
DROP_COLUMNS = {
    # admin / phone
    "Id", "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "QSTVER", "QSTLANG", "MSCODE", "NUMADULT", "NUMMEN",
    "NUMWOMEN", "PVTRESD1", "CELLFON3", "CADULT", "PVTRESD2", "CSTATE",
    "LANDLINE", "NUMHHOL2", "HHADULT",
    # survey weights
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_LLCPWT",
    # complex formats with a clean calculated version
    "WEIGHT2", "HEIGHT3", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN",
    "FVORANG", "VEGETAB1", "ALCDAY5", "EXEROFT1", "EXEROFT2", "STRENGTH",
    "EXERHMM1", "EXERHMM2",
    # activity types, places, dates
    "EXRACT11", "EXRACT21", "IMFVPLAC", "FLSHTMY2", "HIVTSTD3",
}


def clean(x, names, groups):
    """Recode special BRFSS codes and drop useless columns.
    Returns the cleaned array and the list of kept column names."""
    x = x.copy()
    names = list(names)
    idx = {n: i for i, n in enumerate(names)}

    def set_nan(col, values):
        if col in idx:
            j = idx[col]
            x[np.isin(x[:, j], values), j] = np.nan

    def replace(col, old, new):
        if col in idx:
            j = idx[col]
            x[x[:, j] == old, j] = new

    # binary with codes: 7/9 -> NaN, 2 -> 0
    for c in groups["binary with codes (1/2 + 7/9)"]:
        set_nan(c, [7, 9])
        replace(c, 2, 0)

    # binary without codes: 2 -> 0
    for c in groups["binary without codes"]:
        replace(c, 2, 0)

    # counts of days / drinks / children: 88 -> 0, 77/99 -> NaN
    for c in ["PHYSHLTH", "MENTHLTH", "POORHLTH", "CHILDREN",
              "AVEDRNK2", "DRNK3GE5", "MAXDRNKS"]:
        replace(c, 88, 0)
        set_nan(c, [77, 99])

    # small scales: 7/9 -> NaN (exceptions handled below)
    special = {"CHECKUP1", "DIABETE3", "BPHIGH4", "INCOME2"}
    for c in groups["small scale with codes"]:
        if c not in special:
            set_nan(c, [7, 9])

    set_nan("INCOME2", [77, 99])

    set_nan("CHECKUP1", [7, 9])
    replace("CHECKUP1", 8, 5)  # 8 = never -> after "5+ years"

    # DIABETE3 / BPHIGH4: 1 = yes, 2 = only during pregnancy, 3 = no, 4 = borderline
    for c in ["DIABETE3", "BPHIGH4"]:
        if c in idx:
            j = idx[c]
            old = x[:, j].copy()
            new = np.full_like(old, np.nan)
            new[old == 1] = 1
            new[old == 4] = 0.5
            new[np.isin(old, [2, 3])] = 0
            x[:, j] = new

    # hidden codes in "no codes detected"
    set_nan("DIABAGE2", [98, 99])
    set_nan("_AGEG5YR", [14])
    set_nan("_AGE65YR", [3])
    set_nan("DROCDY3_", [900])
    set_nan("_DRNKWEK", [99900])
    set_nan("MAXVO2_", [99900])

    drop = set(groups["mostly missing (>90% NaN)"]) | set(groups["constant"]) | DROP_COLUMNS
    keep = [j for j, n in enumerate(names) if n not in drop]
    kept_names = [names[j] for j in keep]
    return x[:, keep], kept_names


def fit_column_filter(x, max_nan=0.9):
    """Mask of columns to keep after cleaning (computed on x_train only)."""
    nan_ratio = np.isnan(x).mean(axis=0)
    n_unique = np.array([len(np.unique(c[~np.isnan(c)])) for c in x.T])
    return (nan_ratio < max_nan) & (n_unique > 1)


# ---------------------------------------------------------------------
# 3. Train / validation split (same for everyone)
# ---------------------------------------------------------------------
def split_train_val(n, ratio=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(ratio * n)
    return perm[:cut], perm[cut:]


# ---------------------------------------------------------------------
# 4. Fill NaNs + standardize (stats from the training part only)
# ---------------------------------------------------------------------
def fit_preprocessing(x):
    mean = np.nanmean(x, axis=0)
    x_filled = np.where(np.isnan(x), mean, x)
    std = x_filled.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    return mean, std


def apply_preprocessing(x, mean, std):
    x = np.where(np.isnan(x), mean, x)    # fill NaNs with training mean
    x = (x - mean) / std                  # standardize
    return np.c_[np.ones(x.shape[0]), x]  # add bias column


# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
def f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y_true, probs):
    """Threshold on predicted probabilities that maximizes F1."""
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        score = f1(y_true, (probs >= t).astype(int))
        if score > best_score:
            best_t, best_score = t, score
    return best_t, best_score


# ---------------------------------------------------------------------
# 6. Baseline model: regularized logistic regression
# ---------------------------------------------------------------------
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def train_logistic(x, y, gamma=0.1, max_iters=500, lambda_=1e-4, verbose=True):
    w = np.zeros(x.shape[1])
    for it in range(max_iters):
        grad = x.T @ (sigmoid(x @ w) - y) / len(y) + 2 * lambda_ * w
        w -= gamma * grad
        if verbose and it % 100 == 0:
            p = sigmoid(x @ w)
            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            print(f"iter {it}: loss = {loss:.4f}")
    return w
