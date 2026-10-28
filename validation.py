"""
Shared evaluation tools: metrics, F1-optimal threshold, stratified splits,
k-fold cross-validation and final retraining. Any model can be evaluated
through a fit_predict(x_train, y_train, x_eval, seed) -> scores function.
"""

from functools import partial
from pathlib import Path
import importlib
import numpy as np

SEED = 0

__all__ = [
    "SEED",
    "to_binary",
    "predict_labels",
    "binary_metrics",
    "confusion_matrix",
    "f1_score",
    "best_f1_threshold",
    "stratified_k_fold",
    "stratified_train_val_split",
    "save_folds",
    "load_folds",
    "prepare_brfss_pair",
    "make_brfss_preprocessor",
    "evaluate_holdout",
    "cross_validate",
    "print_cv_summary",
    "fit_predict_final",
]


def _integer(value, name, minimum=0):
    """Check that value is an integer >= minimum and return it as int."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _vector(values, name):
    """Check that values is a non-empty 1D numeric vector without NaN/inf."""
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a non-empty 1D numeric vector.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or inf.")
    return a


def to_binary(y):
    """Copy of the labels as 0/1. Accepts 0/1 or -1/+1, but not a mix of both."""
    a = _vector(y, "y")
    if not (np.isin(a, [0, 1]).all() or np.isin(a, [-1, 1]).all()):
        raise ValueError("Expected labels: 0/1 or -1/+1, not a mix of both.")
    return (a == 1).astype(np.int64)


def _scores(scores, n):
    """Check that scores is a numeric vector of length n; return it as floats."""
    a = _vector(scores, "scores").astype(float, copy=False)
    if len(a) != n:
        raise ValueError("scores and labels must have the same length.")
    return a


def _threshold(value):
    """Check that the threshold is a numeric scalar (not NaN); return a float."""
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in "iuf" or np.isnan(a):
        raise ValueError("The threshold must be a numeric scalar, not NaN.")
    return float(a)


def predict_labels(scores, threshold=0.5, negative_label=0):
    """Labels from scores: 1 if score >= threshold, else negative_label.

    Typical thresholds: 0.5 for probabilities, 0 for margins.
    negative_label=-1 gives the submission coding of the project.
    threshold=+inf predicts no positive, threshold=-inf predicts all positive.
    """
    a = _vector(scores, "scores")
    if negative_label not in (0, -1):
        raise ValueError("negative_label must be 0 or -1.")
    return np.where(a >= _threshold(threshold), 1, negative_label)


def binary_metrics(y_true, y_pred):
    """TP/FP/TN/FN, accuracy, precision, recall, specificity and F1.

    The two vectors may use different codings (0/1 or -1/+1). A metric with a
    zero denominator is 0. F1 is the F1 of the positive class (not a macro or
    weighted average).
    """
    y, p = to_binary(y_true), to_binary(y_pred)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fn = int(np.sum((y == 1) & (p == 0)))

    def divide(a, b):
        """a / b, or 0 if b is 0."""
        return float(a / b) if b else 0.0

    return dict(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=divide(tp + tn, len(y)),
        precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn),
        specificity=divide(tn, tn + fp),
        f1=divide(2 * tp, 2 * tp + fp + fn),
    )


def f1_score(y_true, y_pred):
    """F1 score of the positive class."""
    return binary_metrics(y_true, y_pred)["f1"]


def best_f1_threshold(y_true, scores):
    """Return (threshold, F1) maximizing F1, with an exact O(n log n) search.

    Every decision "score >= threshold" is examined, equal scores being treated
    together. On ties, the largest threshold (fewest positives) is chosen.
    Scores outside [0, 1] (margins) are accepted. The returned F1 is optimized
    on these labels, so it is not an independent test score.
    Without any true positive, returns (inf, 0).
    """
    y = to_binary(y_true)
    s = _scores(scores, len(y))
    order = np.argsort(-s, kind="stable")
    ranked = s[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(y) - 1]
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    # 2TP + FP + FN = number predicted positive + number truly positive.
    values = 2.0 * tp / (ends + 1 + int(y.sum()))
    candidates = np.r_[np.inf, ranked[ends]]
    values = np.r_[0.0, values]
    k = int(np.argmax(values))
    return float(candidates[k]), float(values[k])


def stratified_k_fold(y, n_splits=5, seed=SEED):
    """Deterministic stratified k-fold: list of (train_indices, val_indices).

    Every sample is in exactly one validation fold. Fold sizes and class counts
    per fold differ by at most 1. Requires both classes, with at least n_splits
    samples each. Does not change the global random state. The same row order,
    labels, seed and n_splits give exactly the same folds in every notebook.
    """
    y = to_binary(y)
    k = _integer(n_splits, "n_splits", 2)
    rng = np.random.default_rng(_integer(seed, "seed"))
    counts = np.bincount(y, minlength=2)
    if counts.min() < k:
        raise ValueError("Each class must contain at least n_splits samples.")
    fold_id = np.empty(len(y), dtype=np.int64)
    offset = 0
    for c in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == c))
        fold_id[indices] = (np.arange(len(indices)) + offset) % k
        offset = (offset + len(indices)) % k
    return [
        (
            rng.permutation(np.flatnonzero(fold_id != j)),
            rng.permutation(np.flatnonzero(fold_id == j)),
        )
        for j in range(k)
    ]


def stratified_train_val_split(y, val_ratio=0.2, seed=SEED):
    """Stratified train/validation split (rounded per class, at least one sample
    of each class in each part). The actual proportion can differ slightly
    from val_ratio, especially on very small datasets.
    """
    y = to_binary(y)
    if not np.isscalar(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1.")
    rng = np.random.default_rng(_integer(seed, "seed"))
    train, val = [], []
    for c in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == c))
        if len(indices) < 2:
            raise ValueError("At least two samples of each class are needed.")
        n_val = min(
            len(indices) - 1, max(1, int(np.floor(len(indices) * val_ratio + 0.5)))
        )
        val.append(indices[:n_val])
        train.append(indices[n_val:])
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(val))


def _matrix(x):
    """Check that x is a non-empty 2D float matrix without inf (NaN allowed)."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0 or np.isinf(a).any():
        raise ValueError("x must be a non-empty 2D matrix without inf (NaN allowed).")
    return a


def _validate_folds(folds, y):
    """Check that the folds form a valid partition of the samples (each sample
    validated exactly once, both classes in every part); return them as int arrays."""
    result, seen = [], np.zeros(len(y), dtype=int)
    for train, val in folds:
        pair = []
        for indices in (train, val):
            a = np.asarray(indices)
            if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "iu":
                raise ValueError("Indices must be non-empty integer vectors.")
            if np.any(a < 0) or np.any(a >= len(y)) or len(np.unique(a)) != len(a):
                raise ValueError("Indices out of range or duplicated.")
            pair.append(a.astype(np.int64, copy=True))
        train, val = pair
        if np.intersect1d(train, val).size or len(train) + len(val) != len(y):
            raise ValueError(
                "Each training part must be the exact complement of its validation part."
            )
        if len(np.unique(y[train])) != 2 or len(np.unique(y[val])) != 2:
            raise ValueError("Each part must contain both classes.")
        seen[val] += 1
        result.append((train, val))
    if len(result) < 2 or not np.all(seen == 1):
        raise ValueError("Each sample must be validated exactly once, over >= 2 folds.")
    return result


def confusion_matrix(y_true, y_pred):
    """Matrix [[TN, FP], [FN, TP]]; rows = true class, columns = predicted class."""
    m = binary_metrics(y_true, y_pred)
    return np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)


def _ids(sample_ids, n):
    """Check that sample_ids is a vector of n unique integer or string ids."""
    ids = np.asarray(sample_ids)
    if ids.ndim != 1 or len(ids) != n or ids.dtype.kind not in "iuUS":
        raise ValueError("sample_ids must be a vector of integer or string ids.")
    if len(np.unique(ids)) != n:
        raise ValueError("Ids must be unique.")
    return ids


def save_folds(path, folds, y, sample_ids):
    """Save the folds with their labels and ids (never overwrites a file).

    The ids (train_ids from load_csv_data) are required to detect any change
    of row order. NumPy format, without pickle.
    """
    labels = to_binary(y)
    ids = _ids(sample_ids, len(labels))
    partitions = _validate_folds(folds, labels)
    data = dict(
        version=np.array(1),
        labels=labels,
        sample_ids=ids,
        n_splits=np.array(len(partitions)),
    )
    for j, (train, val) in enumerate(partitions):
        data[f"train_{j}"], data[f"val_{j}"] = train, val
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **data)


def load_folds(path, y, sample_ids):
    """Load saved folds and check that the labels and the exact id order match."""
    labels = to_binary(y)
    ids = _ids(sample_ids, len(labels))
    with np.load(path, allow_pickle=False) as data:
        if data["version"].shape != () or int(data["version"]) != 1:
            raise ValueError("Unsupported fold file version.")
        if not np.array_equal(data["labels"], labels) or not np.array_equal(
            data["sample_ids"], ids
        ):
            raise ValueError("Labels or id order differ from the saved folds.")
        k = _integer(data["n_splits"].item(), "n_splits", 2)
        partitions = [(data[f"train_{j}"], data[f"val_{j}"]) for j in range(k)]
    return _validate_folds(partitions, labels)


def _names(names, n):
    """Check that names contains one unique, non-empty name per column."""
    names = list(names)
    if len(names) != n or not all(isinstance(v, str) and v for v in names):
        raise ValueError(
            "names must contain one non-empty name per column, without Id."
        )
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique.")
    return names


def prepare_brfss_pair(
    x_train,
    x_eval,
    *,
    names,
    clean_data=True,
    categorical_columns=(),
    add_missing_indicators=False,
    min_nan=0.05,
    max_nan=0.9,
    interaction_pairs=(),
    square_columns=(),
    standardize=True,
    add_bias=False,
    preprocessing_module=None,
    return_names=False,
):
    """Apply the shared preprocessing.py to a (train, evaluation) pair.

    Everything that is learned uses x_train only:
      1. compute_groups -> clean on both parts (if clean_data=True).
      2. fit_column_filter on train, same mask on the evaluation part.
      3. add_interactions on the cleaned original variables.
      4. nan_indicators BEFORE one_hot and imputation.
      5. one_hot: categories found on train only.
      6. Columns entirely NaN on train are removed, then the shared
         fit_preprocessing / apply_preprocessing (MEAN imputation).

    clean_data=False: no BRFSS cleaning and no 90 % filter; NaNs are imputed
    and entirely missing columns removed. Expansions still follow their options.
    standardize=False: mean imputation only, without centering/scaling.
    add_bias=False suits a neural network with its own biases, and trees.
    add_bias=True suits linear functions without a built-in intercept.

    Interactions come before one-hot and use the original names. A requested
    column that is not in names raises an error; if cleaning removes it in this
    training part, its expansion is skipped (as in the shared module), so the
    number of features can vary between folds. NaN and unseen categories give
    all-zero dummies. Missing indicators follow the NaN ratios of the TRAIN part.

    preprocessing_module=None imports 'preprocessing' at call time.
    Returns (train_transformed, eval_transformed), plus the final names if
    return_names=True. The inputs are not modified.
    """
    a, b = _matrix(x_train).copy(), _matrix(x_eval).copy()
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "Train and evaluation parts must have the same initial columns."
        )
    names = _names(names, a.shape[1])
    categories, squares = list(categorical_columns), list(square_columns)
    pairs = list(interaction_pairs)
    if any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in pairs):
        raise ValueError("Each interaction must be a pair of names.")
    requested = categories + squares + [n for pair in pairs for n in pair]
    if any(not isinstance(n, str) or n not in names for n in requested):
        raise ValueError("A requested expansion refers to an unknown initial column.")
    if len(set(categories)) != len(categories) or len(set(squares)) != len(squares):
        raise ValueError("Duplicated columns in the requested expansions.")
    pair_keys = [tuple(sorted(p)) for p in pairs]
    if len(set(pair_keys)) != len(pair_keys) or any(
        a == b and a in squares for a, b in pairs
    ):
        raise ValueError(
            "Duplicated interactions (including a square and the product of a column with itself)."
        )
    if not (0 <= min_nan <= max_nan <= 1):
        raise ValueError("0 <= min_nan <= max_nan <= 1 is required.")
    module = preprocessing_module
    if module is None:
        module = importlib.import_module("preprocessing")
    required = ["fit_preprocessing", "apply_preprocessing"]
    if clean_data:
        required += ["compute_groups", "clean", "fit_column_filter"]
    if categories:
        required.append("one_hot")
    if add_missing_indicators:
        required.append("nan_indicators")
    if pairs or squares:
        required.append("add_interactions")
    absent = [n for n in required if not callable(getattr(module, n, None))]
    if absent:
        raise ValueError(
            "preprocessing.py is missing these functions: " + ", ".join(absent)
        )
    if clean_data:
        groups = module.compute_groups(a, names)
        a, cleaned_names = module.clean(a, names, groups)
        b, eval_names = module.clean(b, names, groups)
        if cleaned_names != eval_names:
            raise ValueError(
                "Cleaning produced different columns for the train and evaluation parts."
            )
        names = cleaned_names
        if a.shape[1]:
            keep = module.fit_column_filter(a)
            a, b = a[:, keep], b[:, keep]
            names = [n for n, selected in zip(names, keep) if selected]
    if a.shape[1] == 0:
        raise ValueError("Cleaning removed every column of this training part.")
    if pairs or squares:
        a, b, names = module.add_interactions(a, b, names, pairs, squares)
    if add_missing_indicators:
        a, b, names = module.nan_indicators(
            a, b, names, min_nan=min_nan, max_nan=max_nan
        )
    if categories:
        a, b, names = module.one_hot(a, b, names, categories)
    # A product can be entirely NaN even if its factors are not; the shared
    # module uses nanmean, so these columns are removed before fitting it.
    keep = np.any(~np.isnan(a), axis=0)
    a, b = a[:, keep], b[:, keep]
    names = [n for n, selected in zip(names, keep) if selected]
    if not names:
        raise ValueError("No column with an observed value in this training part.")
    _names(names, a.shape[1])
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Numerical overflow in the interactions or the data.")
    mean, std = module.fit_preprocessing(a)
    mean, std = np.asarray(mean), np.asarray(std)
    if (
        mean.shape != (a.shape[1],)
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("Preprocessing produced invalid means / standard deviations.")
    if standardize:
        a, b = module.apply_preprocessing(a, mean, std), module.apply_preprocessing(
            b, mean, std
        )
        # apply_preprocessing adds a column of ones in first position.
        if (
            a.shape[1] != len(names) + 1
            or b.shape[1] != len(names) + 1
            or not np.all(a[:, 0] == 1)
            or not np.all(b[:, 0] == 1)
        ):
            raise ValueError(
                "apply_preprocessing must add a bias column in first position."
            )
        if not add_bias:
            a, b = a[:, 1:], b[:, 1:]
    else:
        a, b = np.where(np.isnan(a), mean, a), np.where(np.isnan(b), mean, b)
        if add_bias:
            a, b = np.c_[np.ones(len(a)), a], np.c_[np.ones(len(b)), b]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("NaN/inf left after preprocessing.")
    final_names = (["__bias__"] if add_bias else []) + names
    return (a, b, final_names) if return_names else (a, b)


def make_brfss_preprocessor(names, **options):
    """Create preprocess(train, eval) to pass to the evaluation functions.

    All statistics are recomputed on the training part of EACH call, so nothing
    fitted is shared between folds. Example:
        preprocess = make_brfss_preprocessor(names, add_bias=False)
    """
    if options.get("return_names", False):
        raise ValueError(
            "The preprocessing callback must return two matrices, without names."
        )
    options = options.copy()
    for key in ("categorical_columns", "square_columns"):
        if key in options:
            options[key] = tuple(options[key])
    if "interaction_pairs" in options:
        options["interaction_pairs"] = tuple(
            tuple(p) for p in options["interaction_pairs"]
        )
    return partial(prepare_brfss_pair, names=tuple(names), **options)


def _validate_xy(x, y):
    """Check x (2D matrix) and y (binary labels) and their lengths."""
    x, y = _matrix(x), to_binary(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows.")
    return x, y


def _threshold_mode(threshold):
    """Return "inner" (threshold tuned inside each fold) or a fixed float."""
    if isinstance(threshold, str) and threshold == "inner":
        return threshold
    return _threshold(threshold)


def _run_model(x_train, y_train, x_eval, fit_predict, preprocess, seed):
    """Preprocess (on copies), train a fresh model with fit_predict and return
    its scores on x_eval."""
    a, b = x_train.copy(), x_eval.copy()
    if preprocess is not None:
        a, b = preprocess(a, b)
    a, b = _matrix(a), _matrix(b)
    if len(a) != len(y_train) or len(b) != len(x_eval) or a.shape[1] != b.shape[1]:
        raise ValueError("Preprocessing changed the rows or misaligned the columns.")
    # NaNs are only allowed if the model explicitly handles them.
    return _scores(fit_predict(a, y_train.copy(), b, seed), len(b)).copy()


def _evaluate_partition(
    x, y, train, val, fit_predict, preprocess, threshold, inner_val_ratio, seeds
):
    """Train on `train`, evaluate on `val`. With threshold="inner", the threshold
    is first tuned on an inner split of `train`. Returns metrics, scores and
    predictions of the fold."""
    chosen, tuning_f1 = threshold, None
    if isinstance(threshold, str):
        it, iv = stratified_train_val_split(y[train], inner_val_ratio, seeds[0])
        inner_scores = _run_model(
            x[train[it]], y[train[it]], x[train[iv]], fit_predict, preprocess, seeds[1]
        )
        chosen, tuning_f1 = best_f1_threshold(y[train[iv]], inner_scores)
    scores = _run_model(x[train], y[train], x[val], fit_predict, preprocess, seeds[2])
    pred = predict_labels(scores, chosen)
    return dict(
        n_train=len(train),
        n_val=len(val),
        threshold=float(chosen),
        inner_tuning_f1=tuning_f1,
        metrics=binary_metrics(y[val], pred),
        scores=scores,
        predictions=pred,
    )


def _seeds(seed, count):
    """`count` independent integer seeds derived from `seed`."""
    sequence = np.random.SeedSequence(_integer(seed, "seed"))
    return [int(c.generate_state(1)[0]) for c in sequence.spawn(count)]


def evaluate_holdout(
    x,
    y,
    fit_predict,
    *,
    split=None,
    val_ratio=0.2,
    seed=SEED,
    preprocess=None,
    threshold=0.5,
    inner_val_ratio=0.2,
):
    """Quick evaluation on a single stratified train/validation split.

    split=(train_indices, val_indices), e.g. folds[0], reuses exactly the same
    holdout; otherwise it is created with val_ratio and seed. Each part must be
    the complement of the other and contain both classes.
    threshold='inner': two trainings, with the threshold tuned internally.
    The result contains metrics, scores, predictions and the original indices.
    """
    x, y = _validate_xy(x, y)
    chosen_mode = _threshold_mode(threshold)
    if split is None:
        split = stratified_train_val_split(y, val_ratio, seed)
    # The two complementary folds let us reuse the fold checks.
    train, val = _validate_folds([split, (split[1], split[0])], y)[0]
    result = _evaluate_partition(
        x,
        y,
        train,
        val,
        fit_predict,
        preprocess,
        chosen_mode,
        inner_val_ratio,
        _seeds(seed, 3),
    )
    result.update(train_indices=train, val_indices=val)
    return result


def cross_validate(
    x,
    y,
    fit_predict,
    *,
    n_splits=5,
    seed=SEED,
    folds=None,
    preprocess=None,
    threshold=0.5,
    inner_val_ratio=0.2,
    verbose=False,
):
    """k-fold cross-validation; the preprocessing is refitted on each training part.

    fit_predict(x_train, y_train_01, x_eval, seed) -> 1D scores, from a new model.
    preprocess(train_raw, eval_raw) -> two matrices; None passes the raw data.
    threshold: fixed scalar, or 'inner' to tune it inside each fold.
    folds: optional shared folds (fully checked).

    Returns the per-fold results, the mean and standard deviation (ddof=0) of
    the metrics, the out-of-fold scores and predictions in the ORIGINAL order,
    and the aggregated out-of-fold metrics. Mean F1 and aggregated F1 differ.

    threshold='inner' costs 2 trainings per fold: the threshold is tuned on an
    inner split, then the model is retrained on the whole training part, so the
    outer score honestly measures this whole procedure.
    """
    x, y = _validate_xy(x, y)
    mode = _threshold_mode(threshold)
    partitions = _validate_folds(
        stratified_k_fold(y, n_splits, seed) if folds is None else folds, y
    )
    seeds = _seeds(seed, 3 * len(partitions))
    oof_scores, oof_pred = np.empty(len(y)), np.empty(len(y), dtype=int)
    oof_fold = np.empty(len(y), dtype=int)
    results = []
    for j, (train, val) in enumerate(partitions):
        result = _evaluate_partition(
            x,
            y,
            train,
            val,
            fit_predict,
            preprocess,
            mode,
            inner_val_ratio,
            seeds[3 * j : 3 * j + 3],
        )
        oof_scores[val] = result.pop("scores")
        oof_pred[val] = result.pop("predictions")
        oof_fold[val] = j
        result["fold"] = j
        results.append(result)
        if verbose:
            print(
                f"Fold {j+1}/{len(partitions)}: F1={result['metrics']['f1']:.4f}, "
                f"threshold={result['threshold']:.6g}",
                flush=True,
            )
    keys = ("accuracy", "precision", "recall", "specificity", "f1")
    mean = {k: float(np.mean([r["metrics"][k] for r in results])) for k in keys}
    std = {k: float(np.std([r["metrics"][k] for r in results])) for k in keys}
    return dict(
        folds=results,
        mean=mean,
        std=std,
        oof_scores=oof_scores,
        oof_predictions=oof_pred,
        oof_fold=oof_fold,
        oof_metrics=binary_metrics(y, oof_pred),
    )


def print_cv_summary(result):
    """Print the per-fold metrics of cross_validate and their mean."""
    print("Fold  F1       Precision  Recall    Accuracy  Threshold")
    for r in result["folds"]:
        m = r["metrics"]
        print(
            f"{r['fold']+1:>3}   {m['f1']:.4f}   {m['precision']:.4f}     "
            f"{m['recall']:.4f}    {m['accuracy']:.4f}    {r['threshold']:.6g}"
        )
    print(f"Mean F1: {result['mean']['f1']:.4f}; std: {result['std']['f1']:.4f}")
    print(f"Aggregated out-of-fold F1: {result['oof_metrics']['f1']:.4f}")


def fit_predict_final(
    x_train,
    y_train,
    x_test,
    fit_predict,
    *,
    preprocess=None,
    threshold=0.5,
    seed=SEED,
    inner_val_ratio=0.2,
    negative_label=-1,
):
    """Retrain on the whole training set and predict the test set (original order).

    threshold: fixed number chosen during development, or 'inner' to choose it
    on an inner holdout before the full retraining. The returned inner F1 is a
    tuning score, never a test F1. No test labels are needed, no file is written.

    Returns scores, predictions (-1/+1 by default), threshold, inner_tuning_f1.
    """
    x_train, y = _validate_xy(x_train, y_train)
    x_test = _matrix(x_test)
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test must have the same initial columns.")
    if negative_label not in (0, -1):
        raise ValueError("negative_label must be 0 or -1.")
    chosen, tuning_f1 = _threshold_mode(threshold), None
    seeds = _seeds(seed, 3)
    if isinstance(chosen, str):
        it, iv = stratified_train_val_split(y, inner_val_ratio, seeds[0])
        s = _run_model(
            x_train[it], y[it], x_train[iv], fit_predict, preprocess, seeds[1]
        )
        chosen, tuning_f1 = best_f1_threshold(y[iv], s)
    scores = _run_model(x_train, y, x_test, fit_predict, preprocess, seeds[2])
    return dict(
        scores=scores,
        predictions=predict_labels(scores, chosen, negative_label),
        threshold=float(chosen),
        inner_tuning_f1=tuning_f1,
    )
