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
        # extra list (not a group of its own): columns whose codes are 77/88/99
        "codes_99_family": [],
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
                if vmax == 99:
                    groups["codes_99_family"].append(name)

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
    "Id",
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "QSTVER",
    "QSTLANG",
    "MSCODE",
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "PVTRESD1",
    "CELLFON3",
    "CADULT",
    "PVTRESD2",
    "CSTATE",
    "LANDLINE",
    "NUMHHOL2",
    "HHADULT",
    # survey weights
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_LLCPWT",
    # complex formats with a clean calculated version
    "WEIGHT2",
    "HEIGHT3",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "ALCDAY5",
    "EXEROFT1",
    "EXEROFT2",
    "STRENGTH",
    "EXERHMM1",
    "EXERHMM2",
    # activity types, places, dates
    "EXRACT11",
    "EXRACT21",
    "IMFVPLAC",
    "FLSHTMY2",
    "HIVTSTD3",
}


def clean(x, names, groups):
    """Recode special BRFSS codes and drop useless columns.
    Returns the cleaned array and the list of kept column names."""
    x = x.copy()
    names = list(names)
    idx = {n: i for i, n in enumerate(names)}

    def set_nan(col, values):
        """Replace the given special codes of column `col` by NaN."""
        if col in idx:
            j = idx[col]
            x[np.isin(x[:, j], values), j] = np.nan

    def replace(col, old, new):
        """Replace the value `old` of column `col` by `new`."""
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
    for c in [
        "PHYSHLTH",
        "MENTHLTH",
        "POORHLTH",
        "CHILDREN",
        "AVEDRNK2",
        "DRNK3GE5",
        "MAXDRNKS",
    ]:
        replace(c, 88, 0)
        set_nan(c, [77, 99])

    # small scales: remove the "don't know / refused" codes
    special = {"CHECKUP1", "DIABETE3", "BPHIGH4"}
    fam99 = set(
        groups.get("codes_99_family", [])
    )  # codes are 77/99 (e.g. INCOME2, _MRACE1, LASTSMK2)
    only_9_is_code = {"EMPLOY1", "_RACE"}  # 7 and 8 are real answers here
    for c in groups["small scale with codes"]:
        if c in special:
            continue
        if c in fam99:
            set_nan(c, [77, 99])
        elif c in only_9_is_code:
            set_nan(c, [9])
        else:
            set_nan(c, [7, 9])

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

    drop = (
        set(groups["mostly missing (>90% NaN)"])
        | set(groups["constant"])
        | DROP_COLUMNS
    )
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
    """Random split of the indices 0..n-1 into a training part (ratio) and a
    validation part (1 - ratio). Returns (train_indices, val_indices)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(ratio * n)
    return perm[:cut], perm[cut:]


# ---------------------------------------------------------------------
# 4. Fill NaNs + standardize (stats from the training part only)
# ---------------------------------------------------------------------
def fit_preprocessing(x):
    """Statistics for imputation and standardization, computed on training data.
    Returns the column means (ignoring NaNs) and the column standard deviations
    after mean imputation (1 for constant columns)."""
    mean = np.nanmean(x, axis=0)
    x_filled = np.where(np.isnan(x), mean, x)
    std = x_filled.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    return mean, std


def apply_preprocessing(x, mean, std):
    """Fill NaNs with the training means, standardize with the training
    statistics and add a bias column of ones in front."""
    x = np.where(np.isnan(x), mean, x)  # fill NaNs with training mean
    x = (x - mean) / std  # standardize
    return np.c_[np.ones(x.shape[0]), x]  # add bias column


# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
def f1(y_true, y_pred):
    """F1 score for labels in {0, 1}: 2 TP / (2 TP + FP + FN)."""
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
    """Logistic function 1 / (1 + exp(-t))."""
    return 1 / (1 + np.exp(-t))


def train_logistic(x, y, gamma=0.1, max_iters=500, lambda_=1e-4, verbose=True):
    """Regularized logistic regression with gradient descent, starting from 0.
    x must already contain the bias column. Prints the loss every 100
    iterations if verbose. Returns the weight vector."""
    w = np.zeros(x.shape[1])
    for it in range(max_iters):
        grad = x.T @ (sigmoid(x @ w) - y) / len(y) + 2 * lambda_ * w
        w -= gamma * grad
        if verbose and it % 100 == 0:
            p = sigmoid(x @ w)
            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            print(f"iter {it}: loss = {loss:.4f}")
    return w


def evaluate(x_feat, y01):
    """Quick check of a feature set: train the logistic regression on 80 % of
    the data, print the validation F1 and accuracy at the best threshold, and
    return the validation F1."""
    i_tr, i_val = split_train_val(len(x_feat))

    mean, std = fit_preprocessing(x_feat[i_tr])
    x_tr = apply_preprocessing(x_feat[i_tr], mean, std)
    x_val = apply_preprocessing(x_feat[i_val], mean, std)

    w = train_logistic(x_tr, y01[i_tr], verbose=False)

    t, score = best_threshold(y01[i_val], sigmoid(x_val @ w))

    y_pred = (sigmoid(x_val @ w) >= t).astype(int)
    acc = np.mean(y_pred == y01[i_val])
    print(
        f"F1 (validation) = {score:.3f}  accuracy = {acc:.3f}  "
        f"with threshold {t:.2f}  ({x_feat.shape[1]} columns)"
    )

    return score


def one_hot(x_train, x_test, names, columns):
    """
    Replace each column in `columns` by one 0/1 column per category.
    Categories are found on x_train only, then applied to x_test.
    Returns new x_train, new x_test, new list of names.
    """
    names = list(names)
    new_tr, new_te, new_names = [], [], []
    to_delete = []

    for c in columns:
        if c not in names:
            continue
        j = names.index(c)
        col = x_train[:, j]
        categories = np.unique(col[~np.isnan(col)])  # found on x_train only

        for v in categories:
            new_tr.append((col == v).astype(float))
            new_te.append((x_test[:, j] == v).astype(float))
            new_names.append(f"{c}={v:g}")

        to_delete.append(j)

    x_train = np.delete(x_train, to_delete, axis=1)
    x_test = np.delete(x_test, to_delete, axis=1)

    if new_tr:  # only if at least one column was transformed
        x_train = np.column_stack([x_train] + new_tr)
        x_test = np.column_stack([x_test] + new_te)

    final_names = [n for i, n in enumerate(names) if i not in to_delete] + new_names
    return x_train, x_test, final_names


def nan_indicators(x_train, x_test, names, min_nan=0.05, max_nan=0.9):
    """
    For each column with between min_nan and max_nan missing values
    (computed on x_train), add a 0/1 column "was missing".
    Returns new x_train, new x_test, new list of names.
    """
    new_tr, new_te, new_names = [], [], []

    for j, c in enumerate(names):
        nan_col = np.isnan(x_train[:, j]).astype(float)
        proportion_nan = nan_col.mean()

        if min_nan <= proportion_nan <= max_nan:
            new_tr.append(nan_col)
            new_te.append(np.isnan(x_test[:, j]).astype(float))
            new_names.append(f"{c}_missing")

    if new_tr:
        x_train = np.column_stack([x_train] + new_tr)
        x_test = np.column_stack([x_test] + new_te)

    final_names = list(names) + new_names
    return x_train, x_test, final_names


def add_interactions(x_train, x_test, names, pairs, squares):
    """
    Add the product of each pair of columns and the square of some columns.
    Returns new x_train, new x_test, new list of names.
    """
    names = list(names)
    new_tr, new_te, new_names = [], [], []

    for a, b in pairs:
        if a in names and b in names:
            ia = names.index(a)
            ib = names.index(b)
            new_tr.append(x_train[:, ia] * x_train[:, ib])
            new_te.append(x_test[:, ia] * x_test[:, ib])
            new_names.append(f"{a}*{b}")

    for a in squares:
        if a in names:
            ia = names.index(a)
            new_tr.append(x_train[:, ia] ** 2)
            new_te.append(x_test[:, ia] ** 2)
            new_names.append(f"{a}^2")

    if new_tr:
        x_train = np.column_stack([x_train] + new_tr)
        x_test = np.column_stack([x_test] + new_te)

    final_names = list(names) + new_names
    return x_train, x_test, final_names


# ---------------------------------------------------------------------
# 7. Final feature pipeline (used by run.py and the notebooks)
# ---------------------------------------------------------------------
# categorical columns without a natural order -> one-hot encoding
CATEGORICAL = [
    "MARITAL",
    "EMPLOY1",
    "RENTHOM1",
    "PERSDOC2",
    "_RACE",
    "SXORIENT",
    "TRNSGNDR",
]

# products of two features and squares (see exploration.ipynb, part 2)
INTERACTIONS = [
    ("_AGE80", "BPHIGH4"),  # age x high blood pressure
    ("_AGE80", "DIABETE3"),  # age x diabetes
    ("_AGE80", "GENHLTH"),  # age x general health
    ("_AGE80", "SEX"),  # age x sex
    ("_AGE80", "CVDSTRK3"),  # age x stroke
    ("GENHLTH", "DIFFWALK"),  # general health x difficulty walking
]
SQUARES = ["_AGE80", "_BMI5"]


def build_features(x_train, x_test, names):
    """Cleaning + column filter + one-hot + interactions.
    Every rule is learned on x_train only, then applied identically to x_test."""
    groups = compute_groups(x_train, names)
    x_tr, kept = clean(x_train, names, groups)
    x_te, _ = clean(x_test, names, groups)

    keep = fit_column_filter(x_tr)  # drop >90 % NaN and constant columns
    x_tr, x_te = x_tr[:, keep], x_te[:, keep]
    kept = [n for n, k in zip(kept, keep) if k]

    x_tr, x_te, kept = one_hot(x_tr, x_te, kept, CATEGORICAL)
    x_tr, x_te, kept = add_interactions(x_tr, x_te, kept, INTERACTIONS, SQUARES)
    return x_tr, x_te


def drop_duplicate_columns(x_train, x_test):
    """Remove columns identical to an earlier one (checked on x_train, NaN == NaN).
    Identical columns (e.g. _CRACE1 == _CPRACE) make X^T X singular."""
    filled = np.nan_to_num(x_train, nan=-12345.678)
    _, first = np.unique(filled, axis=1, return_index=True)
    first = np.sort(first)
    return x_train[:, first], x_test[:, first]


def best_threshold_quantiles(y_true, scores):
    """Threshold that maximizes F1. The candidates are 100 quantiles of the
    scores (50 % to 99.5 %), so it works for any score scale."""
    best_t, best_f = 0.0, -1.0
    for t in np.quantile(scores, np.linspace(0.50, 0.995, 100)):
        score = f1(y_true, (scores >= t).astype(int))
        if score > best_f:
            best_t, best_f = t, score
    return best_t, best_f
