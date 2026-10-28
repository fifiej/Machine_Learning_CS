"""
Neural network for binary classification, NumPy only.

Part 1 - the model (BinaryMLP):
    one hidden ReLU layer, weighted binary cross-entropy, L2 penalty, Adam.
    hidden_size=0 gives a logistic-regression baseline. All random seeds are
    local to the model. The gradients are those of
        mean(weight * BCE) + l2 / 2 * sum(weights ** 2)   (biases not penalized).
    With a positive weight != 1 the sigmoid output is a score to threshold,
    not a calibrated probability.

Part 2 - the reproducible experiment (tuning, holdout evaluation, submission):
    python neural_network.py --project-dir . --cache-dir work/nn_cache --result-dir nn_results
    The holdout set (folds[0]) is used neither for the architecture choice, nor
    for early stopping, nor for the threshold / ensemble choice: it is evaluated
    once at the end. The submission CSV is only written locally.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from validation import (
    to_binary,
    stratified_k_fold,
    stratified_train_val_split,
    prepare_brfss_pair,
    best_f1_threshold,
    binary_metrics,
)


def sigmoid(z):
    """Numerically stable logistic function."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def _labels(y, n):
    """Check that y is a 0/1 vector of length n."""
    y = np.asarray(y)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must be a 0/1 vector aligned with x.")
    return y


class BinaryMLP:
    """Neural network with one hidden ReLU layer and a sigmoid output.

    The inputs must already be imputed and standardized; clip=10 bounds them.
    Adam is trained on mini-batches shuffled at every epoch. fit() always
    re-initializes the weights, so nothing is shared between two folds.
    """

    def __init__(
        self,
        hidden_size=32,
        learning_rate=1e-3,
        l2=1e-4,
        positive_weight=1.0,
        batch_size=512,
        epochs=25,
        seed=0,
        clip=10.0,
        dtype="float32",
    ):
        """Check and store the hyperparameters (the weights are created by fit)."""
        for name, value, lower in [
            ("hidden_size", hidden_size, 0),
            ("batch_size", batch_size, 1),
            ("epochs", epochs, 1),
            ("seed", seed, 0),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < lower
            ):
                raise ValueError(f"{name} must be an integer >= {lower}.")
        for name, value in [
            ("learning_rate", learning_rate),
            ("positive_weight", positive_weight),
            ("clip", clip),
        ]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be strictly positive and finite.")
        if not np.isfinite(l2) or l2 < 0 or dtype not in ("float32", "float64"):
            raise ValueError("l2 must be >= 0 and dtype float32 or float64.")
        self.config = dict(
            hidden_size=int(hidden_size),
            learning_rate=float(learning_rate),
            l2=float(l2),
            positive_weight=float(positive_weight),
            batch_size=int(batch_size),
            epochs=int(epochs),
            seed=int(seed),
            clip=float(clip),
            dtype=dtype,
        )
        self.params = None

    def _x(self, x):
        """Check the input matrix, convert it to the model dtype and clip it."""
        a = np.asarray(x, dtype=self.config["dtype"])
        if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
            raise ValueError(
                "The network expects a finite, imputed and standardized matrix."
            )
        if self.params is not None and a.shape[1] != self.n_features_:
            raise ValueError("Number of features incompatible with the network.")
        return np.clip(a, -self.config["clip"], self.config["clip"])

    def _initialize(self, n_features, prevalence):
        """Create the weights (He initialization) and return the random generator."""
        rng = np.random.default_rng(self.config["seed"])
        h = self.config["hidden_size"]
        dtype = self.config["dtype"]
        # start the output bias at the optimum of a constant (weighted) prediction
        p = np.clip(prevalence, 1e-6, 1 - 1e-6)
        bias = np.log(p * self.config["positive_weight"] / (1 - p))
        if h:
            self.params = dict(
                w1=(rng.normal(size=(n_features, h)) * np.sqrt(2 / n_features)).astype(
                    dtype
                ),
                b1=np.zeros(h, dtype=dtype),
                w2=(rng.normal(size=h) * 0.1 / np.sqrt(h)).astype(dtype),
                b2=np.array([bias], dtype=dtype),
            )
        else:
            self.params = dict(
                w2=np.zeros(n_features, dtype=dtype), b2=np.array([bias], dtype=dtype)
            )
        self.n_features_ = n_features
        return rng

    def _forward(self, x):
        """Return the output logits and the hidden activations."""
        p = self.params
        if self.config["hidden_size"]:
            hidden = np.maximum(x @ p["w1"] + p["b1"], 0)
        else:
            hidden = x
        return hidden @ p["w2"] + p["b2"][0], hidden

    def _loss_grad(self, x, y):
        """Loss (with L2 penalty) and gradients on a mini-batch x (already prepared)."""
        z, hidden = self._forward(x)
        weight = np.where(y == 1, self.config["positive_weight"], 1.0).astype(x.dtype)
        loss = np.mean(
            weight * ((1 - y) * np.logaddexp(0.0, z) + y * np.logaddexp(0.0, -z))
        )
        delta = (sigmoid(z) - y) * weight / len(y)
        l2 = self.config["l2"]
        gradients = dict(
            w2=hidden.T @ delta + l2 * self.params["w2"],
            b2=np.array([delta.sum()], dtype=x.dtype),
        )
        if self.config["hidden_size"]:
            dh = delta[:, None] * self.params["w2"][None, :]
            dh *= hidden > 0
            gradients["w1"] = x.T @ dh + l2 * self.params["w1"]
            gradients["b1"] = dh.sum(axis=0)
        penalty = sum(
            float(np.sum(v * v)) for k, v in self.params.items() if k.startswith("w")
        )
        return float(loss + 0.5 * l2 * penalty), gradients

    def predict_scores(self, x, batch_size=8192):
        """Sigmoid output for every row of x, computed by batches."""
        if self.params is None:
            raise ValueError("Train or load the network before predict_scores.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        a = self._x(x)
        out = np.empty(len(a), dtype=float)
        for start in range(0, len(a), batch_size):
            z, _ = self._forward(a[start : start + batch_size])
            out[start : start + batch_size] = sigmoid(z)
        if not np.isfinite(out).all():
            raise FloatingPointError(
                "Non-finite scores: lower learning_rate or inspect the data."
            )
        return out

    def fit(self, x, y, *, monitor=None, patience=5, min_delta=1e-4, verbose=False):
        """Train the network; monitor(model) -> dict containing 'selection_score'.

        The monitor must only use the tuning set. The kept weights are those
        of the epoch with the best selection_score (early stopping with
        `patience`). Without a monitor, all epochs are run. The training
        history and the best epoch are stored in history_ and best_epoch_.
        """
        if (
            not isinstance(patience, (int, np.integer))
            or patience < 1
            or not np.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError("patience >= 1 and min_delta >= 0 are required.")
        # every call to fit trains a new network from scratch
        self.params = None
        x = self._x(x)
        y = _labels(y, len(x)).astype(x.dtype)
        if len(np.unique(y)) != 2:
            raise ValueError("Both classes must be present in the training data.")
        rng = self._initialize(x.shape[1], float(y.mean()))
        first = {k: np.zeros_like(v) for k, v in self.params.items()}
        second = {k: np.zeros_like(v) for k, v in self.params.items()}
        step, wait, best_score, best = 0, 0, -np.inf, None
        significant_best = -np.inf
        self.history_ = []
        self.best_epoch_ = self.config["epochs"]
        for epoch in range(1, self.config["epochs"] + 1):
            permutation = rng.permutation(len(y))
            loss_sum = 0.0
            for start in range(0, len(y), self.config["batch_size"]):
                idx = permutation[start : start + self.config["batch_size"]]
                loss, gradients = self._loss_grad(x[idx], y[idx])
                if not np.isfinite(loss):
                    raise FloatingPointError("Non-finite loss.")
                loss_sum += loss * len(idx)
                step += 1
                for k, grad in gradients.items():
                    if not np.isfinite(grad).all():
                        raise FloatingPointError(f"Non-finite gradient: {k}.")
                    first[k] *= 0.9
                    first[k] += 0.1 * grad
                    second[k] *= 0.999
                    second[k] += 0.001 * grad * grad
                    m = first[k] / (1 - 0.9**step)
                    v = second[k] / (1 - 0.999**step)
                    self.params[k] -= (
                        self.config["learning_rate"] * m / (np.sqrt(v) + 1e-8)
                    )
            row = dict(epoch=epoch, training_batch_loss=loss_sum / len(y))
            if monitor is not None:
                metrics = monitor(self)
                score = float(metrics["selection_score"])
                if not np.isfinite(score):
                    raise ValueError("selection_score must be finite.")
                row.update(metrics)
                if score > best_score:
                    best_score, self.best_epoch_ = score, epoch
                    best = {k: v.copy() for k, v in self.params.items()}
                if score > significant_best + min_delta:
                    significant_best, wait = score, 0
                else:
                    wait += 1
            self.history_.append(row)
            if verbose:
                suffix = (
                    f", tuning_F1={row['selection_score']:.4f}"
                    if monitor is not None
                    else ""
                )
                print(
                    f"epoch {epoch:02d}: batch_loss={row['training_batch_loss']:.5f}{suffix}",
                    flush=True,
                )
            if monitor is not None and wait >= patience:
                break
        if best is not None:
            self.params = best
        return self

    def save(self, path):
        """Save the weights and the configuration (NumPy format, without pickle)."""
        if self.params is None:
            raise ValueError("No weights to save.")
        metadata = dict(
            config=self.config,
            n_features=self.n_features_,
            best_epoch=self.best_epoch_,
            history=self.history_,
        )
        with Path(path).open("wb") as f:
            np.savez_compressed(
                f, metadata=np.array(json.dumps(metadata)), **self.params
            )

    @classmethod
    def load(cls, path):
        """Load a network saved with save()."""
        with np.load(path, allow_pickle=False) as f:
            metadata = json.loads(f["metadata"].item())
            obj = cls(**metadata["config"])
            obj.n_features_ = int(metadata["n_features"])
            h = obj.config["hidden_size"]
            shapes = {"w2": (h or obj.n_features_,), "b2": (1,)}
            if h:
                shapes.update(w1=(obj.n_features_, h), b1=(h,))
            obj.params = {k: f[k].copy() for k in shapes}
            if any(
                obj.params[k].shape != shape or not np.isfinite(obj.params[k]).all()
                for k, shape in shapes.items()
            ):
                raise ValueError("Invalid checkpoint.")
            obj.best_epoch_ = int(metadata["best_epoch"])
            obj.history_ = metadata["history"]
            return obj


# =====================================================================
# Experiment: tuning, holdout evaluation and submission
# =====================================================================

SEED = 0
PREPROCESS_OPTIONS = dict(
    categorical_columns=["SEX", "MARITAL", "EMPLOY1"],
    add_missing_indicators=True,
    standardize=True,
    add_bias=False,
)
CANDIDATES = [
    dict(
        name="linear_adam",
        hidden_size=0,
        learning_rate=0.003,
        l2=0.001,
        positive_weight=1.0,
        epochs=25,
    ),
    dict(
        name="nn32",
        hidden_size=32,
        learning_rate=0.001,
        l2=0.001,
        positive_weight=1.0,
        epochs=30,
    ),
    dict(
        name="nn64",
        hidden_size=64,
        learning_rate=0.001,
        l2=0.003,
        positive_weight=1.0,
        epochs=30,
    ),
    dict(
        name="nn32_weighted",
        hidden_size=32,
        learning_rate=0.001,
        l2=0.003,
        positive_weight=3.0,
        epochs=30,
    ),
    dict(
        name="nn64_low_l2",
        hidden_size=64,
        learning_rate=0.001,
        l2=0.0001,
        positive_weight=1.0,
        epochs=35,
    ),
    dict(
        name="nn128_low_l2",
        hidden_size=128,
        learning_rate=0.0005,
        l2=0.0001,
        positive_weight=1.0,
        epochs=35,
    ),
    dict(
        name="nn64_weighted",
        hidden_size=64,
        learning_rate=0.001,
        l2=0.0003,
        positive_weight=3.0,
        epochs=35,
    ),
]


def write_json(path, value):
    """Write value as an indented JSON file."""
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_project_module(project_dir, filename, module_name):
    """Import a Python file of the project under the given module name."""
    path = Path(project_dir) / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_shared(project_dir):
    """Import the shared preprocessing.py of the project."""
    return read_project_module(
        project_dir, "preprocessing.py", "brfss_nn_preprocessing"
    )


def file_hash(path):
    """SHA-256 fingerprint of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv_features(path, cache_dir):
    """Read a CSV row by row (less memory than genfromtxt) and cache it as .npy.

    The cache is invalidated when the path, size or modification time of the
    CSV changes. It only contains copies of the provided CSV files.
    Returns (features, ids, column names).
    """
    path = Path(path).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = cache_dir / path.stem
    meta_path = prefix.with_suffix(".json")
    feature_path = prefix.with_suffix(".npy")
    ids_path = cache_dir / (path.stem + "_ids.npy")
    stat = path.stat()
    signature = dict(source=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if meta_path.exists() and feature_path.exists() and ids_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["signature"] == signature:
            return (
                np.load(feature_path, mmap_mode="r"),
                np.load(ids_path),
                meta["names"],
            )
    print(f"Reading and caching {path.name}...", flush=True)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n = sum(1 for _ in reader)
    if header[0] != "Id":
        raise ValueError("The first column must be Id.")
    x = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype="float64", shape=(n, len(header) - 1)
    )
    ids = np.empty(n, dtype=np.int64)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV line: {i+2}")
            ids[i] = int(row[0])
            x[i] = np.asarray(
                [value if value else "nan" for value in row[1:]], dtype=float
            )
    x.flush()
    del x
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicated ids.")
    np.save(ids_path, ids)
    write_json(meta_path, dict(signature=signature, names=header[1:]))
    return np.load(feature_path, mmap_mode="r"), ids, header[1:]


def load_training_data(project_dir, cache_dir):
    """Load x_train (cached), the 0/1 labels, the ids and the column names."""
    data = Path(project_dir) / "data" / "dataset"
    x, ids, names = _csv_features(data / "x_train.csv", cache_dir)
    targets = np.loadtxt(
        data / "y_train.csv", delimiter=",", skiprows=1, dtype=np.int64
    )
    if targets.shape != (len(x), 2) or not np.array_equal(ids, targets[:, 0]):
        raise ValueError("The ids of the features and of the labels are not aligned.")
    return x, to_binary(targets[:, 1]), ids, names


def prepare(xa, xb, names, shared):
    """Network features (PREPROCESS_OPTIONS) learned on xa and applied to xb."""
    a, b = prepare_brfss_pair(
        xa, xb, names=names, preprocessing_module=shared, **PREPROCESS_OPTIONS
    )
    # float32 for the network products; the shared fit stays in float64
    return a.astype(np.float32), b.astype(np.float32)


def tune_models(x, y, names, shared, result_dir, candidates=None):
    """Train every candidate architecture, pick the best network on the tuning
    set (single network or 3-seed ensemble) and save the frozen selection."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    development, holdout = stratified_k_fold(y, seed=SEED)[0]
    local_train, local_tune = stratified_train_val_split(
        y[development], val_ratio=0.2, seed=SEED + 1
    )
    train, tuning = development[local_train], development[local_tune]
    np.savez_compressed(
        result_dir / "experiment_splits.npz",
        train=train,
        tuning=tuning,
        development=development,
        holdout=holdout,
    )
    print(
        f"Train={len(train)}, tuning={len(tuning)}, reserved holdout={len(holdout)}",
        flush=True,
    )
    a, b = prepare(x[train], x[tuning], names, shared)
    print(f"Network features: {a.shape[1]}", flush=True)
    models, scores, rows = [], [], []

    def train_candidate(config, seed, label):
        """Train one candidate with early stopping and record its tuning metrics."""
        params = {k: v for k, v in config.items() if k != "name"}
        model = BinaryMLP(**params, seed=seed)

        def monitor(m):
            """Tuning F1 at the best threshold, used for early stopping."""
            s = m.predict_scores(b)
            threshold, f1 = best_f1_threshold(y[tuning], s)
            metrics = binary_metrics(y[tuning], (s >= threshold).astype(int))
            return dict(
                selection_score=f1, threshold=threshold, accuracy=metrics["accuracy"]
            )

        start = time.monotonic()
        print(f"\nTrial {label}", flush=True)
        model.fit(a, y[train], monitor=monitor, patience=5, verbose=True)
        s = model.predict_scores(b)
        threshold, f1 = best_f1_threshold(y[tuning], s)
        row = dict(
            name=label,
            config=model.config,
            best_epoch=model.best_epoch_,
            threshold=threshold,
            tuning=binary_metrics(y[tuning], (s >= threshold).astype(int)),
            tuning_at_05=binary_metrics(y[tuning], (s >= 0.5).astype(int)),
            seconds=time.monotonic() - start,
            checkpoint=f"{label}.npz",
        )
        model.save(result_dir / row["checkpoint"])
        write_json(result_dir / f"{label}_history.json", model.history_)
        models.append(model)
        scores.append(s)
        rows.append(row)
        write_json(result_dir / "tuning_trials.json", rows)
        return row

    for candidate in (CANDIDATES if candidates is None else candidates):
        train_candidate(candidate, SEED, candidate["name"])
    # select a network; the linear candidate is only kept as a reference
    neural = [i for i, r in enumerate(rows) if r["config"]["hidden_size"] > 0]
    if not neural:
        raise ValueError("At least one network candidate is required.")
    best = max(neural, key=lambda i: rows[i]["tuning"]["f1"])
    config = rows[best]["config"].copy()
    config.pop("seed")
    selected = [best]
    for seed in (1, 2):
        train_candidate(config, seed, f"{rows[best]['name']}_seed{seed}")
        selected.append(len(rows) - 1)
    ensemble_scores = np.mean([scores[i] for i in selected], axis=0)
    ensemble_t, ensemble_f1 = best_f1_threshold(y[tuning], ensemble_scores)
    best_single = max(
        neural + [len(rows) - 2, len(rows) - 1], key=lambda i: rows[i]["tuning"]["f1"]
    )
    # all these choices only use the tuning set, never the holdout
    if ensemble_f1 > rows[best_single]["tuning"]["f1"]:
        choice = selected
        threshold = ensemble_t
        tuning_metrics = binary_metrics(
            y[tuning], (ensemble_scores >= threshold).astype(int)
        )
        kind = "ensemble_three_seeds"
    else:
        choice = [best_single]
        threshold = rows[best_single]["threshold"]
        tuning_metrics = rows[best_single]["tuning"]
        kind = "single_network"
    members = []
    for i in choice:
        config = rows[i]["config"].copy()
        config["epochs"] = rows[i]["best_epoch"]
        members.append(dict(name=rows[i]["name"], config=config))
    selection = dict(
        kind=kind,
        members=members,
        threshold=threshold,
        tuning_metrics=tuning_metrics,
        preprocess_options=PREPROCESS_OPTIONS,
        selection_rule="maximum tuning F1; accuracy reported, not a selection criterion",
        outer_seed=SEED,
        inner_seed=SEED + 1,
        sample_counts=dict(train=len(train), tuning=len(tuning), holdout=len(holdout)),
        ensemble_tuning_f1=ensemble_f1,
    )
    # the selection is saved before any measure on the holdout
    write_json(result_dir / "selection.json", selection)
    print(
        f"\nFrozen selection: {kind}, tuning F1={tuning_metrics['f1']:.4f}, threshold={threshold:.6f}",
        flush=True,
    )
    return selection


def train_fixed_members(a, y, b, selection, result_dir, prefix):
    """Retrain the selected network(s) on a with their fixed number of epochs
    and return the average score on b."""
    aggregate = np.zeros(len(b), dtype=float)
    for member in selection["members"]:
        print(
            f"Retraining {prefix}: {member['name']}, {member['config']['epochs']} epochs",
            flush=True,
        )
        model = BinaryMLP(**member["config"]).fit(a, y, verbose=True)
        model.save(Path(result_dir) / f"{prefix}_{member['name']}.npz")
        aggregate += model.predict_scores(b) / len(selection["members"])
    return aggregate


def evaluate_frozen(x, y, names, shared, result_dir, selection):
    """Evaluate the frozen selection once on the holdout set and save the metrics."""
    result_dir = Path(result_dir)
    if (result_dir / "holdout_metrics.json").exists():
        raise FileExistsError(
            "The holdout was already evaluated in this folder; read the existing result."
        )
    splits = np.load(result_dir / "experiment_splits.npz", allow_pickle=False)
    development, holdout = splits["development"], splits["holdout"]
    a, b = prepare(x[development], x[holdout], names, shared)
    scores = train_fixed_members(a, y[development], b, selection, result_dir, "holdout")
    pred = (scores >= selection["threshold"]).astype(int)
    metrics = dict(
        selected_threshold=binary_metrics(y[holdout], pred),
        at_05=binary_metrics(y[holdout], (scores >= 0.5).astype(int)),
        always_negative=binary_metrics(y[holdout], np.zeros(len(holdout), dtype=int)),
        threshold=selection["threshold"],
        n=len(holdout),
        note="Configuration, epochs, ensemble and threshold were frozen before this evaluation.",
    )
    np.savez_compressed(
        result_dir / "holdout_predictions.npz",
        indices=holdout,
        y=y[holdout],
        scores=scores,
        predictions=pred,
    )
    write_json(result_dir / "holdout_metrics.json", metrics)
    print("\nHOLDOUT EVALUATION", json.dumps(metrics, indent=2), flush=True)
    return metrics


def generate_submission(
    project_dir, cache_dir, result_dir, x, y, ids, names, shared, helpers, selection
):
    """Retrain on all the training data, predict x_test and write submission_nn.csv."""
    result_dir = Path(result_dir)
    xt, test_ids, test_names = _csv_features(
        Path(project_dir) / "data" / "dataset" / "x_test.csv", cache_dir
    )
    if test_names != names:
        raise ValueError("Train and test columns differ.")
    a, b = prepare(x, xt, names, shared)
    scores = train_fixed_members(a, y, b, selection, result_dir, "final")
    pred = np.where(scores >= selection["threshold"], 1, -1)
    if len(pred) != len(test_ids) or not np.isin(pred, [-1, 1]).all():
        raise ValueError("Invalid final predictions.")
    output = result_dir / "submission_nn.csv"
    # official helper provided with the project
    helpers.create_csv_submission(test_ids, pred, str(output))
    # re-read the written file to check its content
    loaded = np.loadtxt(output, delimiter=",", skiprows=1, dtype=np.int64)
    if (
        loaded.shape != (len(test_ids), 2)
        or not np.array_equal(loaded[:, 0], test_ids)
        or not np.array_equal(loaded[:, 1], pred)
    ):
        raise AssertionError("The written CSV does not match the ids / predictions.")
    np.save(result_dir / "test_scores.npy", scores)
    write_json(
        result_dir / "submission_info.json",
        dict(
            rows=len(pred),
            positives=int(np.sum(pred == 1)),
            threshold=selection["threshold"],
            sha256=file_hash(output),
            submitted=False,
        ),
    )
    print(f"CSV ready: {output}", flush=True)
    return output


def run_single_baseline(project_dir, cache_dir, result_dir, baseline_config=None):
    """Single MLP baseline: split -> threshold / early stopping on the tuning set
    -> holdout evaluation -> submission CSV.

    baseline_config can override hidden_size, learning_rate, l2,
    positive_weight, batch_size, epochs, seed, clip and dtype.
    """
    project_dir = Path(project_dir).resolve()
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir / "selection.json").exists():
        raise FileExistsError(
            "An experiment already exists in this folder; use a new result folder."
        )
    shared = read_shared(project_dir)
    helpers = read_project_module(project_dir, "helpers.py", "brfss_nn_helpers")
    x, y, ids, names = load_training_data(project_dir, cache_dir)
    folds = stratified_k_fold(y, seed=SEED)
    development, holdout = folds[0]
    local_train, local_tune = stratified_train_val_split(
        y[development], val_ratio=0.2, seed=SEED + 1
    )
    train, tuning = development[local_train], development[local_tune]
    splits = dict(train=train, tuning=tuning, development=development, holdout=holdout)
    np.savez_compressed(result_dir / "experiment_splits.npz", **splits)
    # simple baseline: one network, fixed hyperparameters, no ensemble
    config = dict(
        name="mlp32_baseline",
        hidden_size=32,
        learning_rate=0.001,
        l2=0.001,
        positive_weight=1.0,
        batch_size=512,
        epochs=30,
        seed=SEED,
        clip=10.0,
        dtype="float32",
    )
    if baseline_config is not None:
        allowed = {
            "hidden_size",
            "learning_rate",
            "l2",
            "positive_weight",
            "batch_size",
            "epochs",
            "seed",
            "clip",
            "dtype",
        }
        unknown = set(baseline_config) - allowed
        if unknown:
            raise ValueError(f"Unknown baseline parameters: {sorted(unknown)}")
        config.update(baseline_config)
    config["name"] = f"mlp{config['hidden_size']}_baseline"
    a, b = prepare(x[train], x[tuning], names, shared)
    model = BinaryMLP(**{k: v for k, v in config.items() if k != "name"})

    def monitor(candidate):
        """Tuning F1 at the best threshold, used for early stopping."""
        score = candidate.predict_scores(b)
        threshold, f1 = best_f1_threshold(y[tuning], score)
        metrics = binary_metrics(y[tuning], (score >= threshold).astype(int))
        return dict(
            selection_score=f1, threshold=threshold, accuracy=metrics["accuracy"]
        )

    print(
        f"Single baseline: MLP with {config['hidden_size']} hidden units; train={len(train)}, tuning={len(tuning)}, holdout={len(holdout)}",
        flush=True,
    )
    model.fit(a, y[train], monitor=monitor, patience=5, verbose=True)
    scores = model.predict_scores(b)
    threshold, tuning_f1 = best_f1_threshold(y[tuning], scores)
    config["epochs"] = model.best_epoch_
    config["name"] = "mlp32_baseline"
    row = dict(
        name=config["name"],
        config=config,
        best_epoch=model.best_epoch_,
        threshold=threshold,
        tuning=binary_metrics(y[tuning], (scores >= threshold).astype(int)),
        tuning_at_05=binary_metrics(y[tuning], (scores >= 0.5).astype(int)),
        checkpoint="mlp32_baseline.npz",
    )
    model.save(result_dir / row["checkpoint"])
    write_json(result_dir / "mlp32_baseline_history.json", model.history_)
    write_json(result_dir / "tuning_trials.json", [row])
    selection = dict(
        kind="single_neural_network_baseline",
        members=[
            dict(
                name=config["name"],
                config={k: v for k, v in config.items() if k != "name"},
            )
        ],
        threshold=threshold,
        tuning_metrics=row["tuning"],
        preprocess_options=PREPROCESS_OPTIONS,
        selection_rule="one fixed architecture (32 hidden units); early stopping and threshold use tuning data only",
        outer_seed=SEED,
        inner_seed=SEED + 1,
        sample_counts=dict(train=len(train), tuning=len(tuning), holdout=len(holdout)),
    )
    manifest = dict(
        project_dir=str(project_dir),
        numpy_version=np.__version__,
        shared_hashes={
            name: file_hash(project_dir / name)
            for name in ("preprocessing.py", "validation.py", "helpers.py")
        },
        source_hashes={
            p: file_hash(Path(__file__).parent / p) for p in ("neural_network.py",)
        },
        rows=len(y),
        positive_count=int(y.sum()),
        feature_names=names,
        train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        data_signatures={
            p: dict(
                size=(project_dir / "data" / "dataset" / p).stat().st_size,
                mtime_ns=(project_dir / "data" / "dataset" / p).stat().st_mtime_ns,
            )
            for p in ("x_train.csv", "y_train.csv", "x_test.csv")
        },
    )
    write_json(result_dir / "manifest.json", manifest)
    write_json(result_dir / "selection.json", selection)
    print(
        f"Frozen threshold={threshold:.6f}; tuning F1={tuning_f1:.4f}; epoch={model.best_epoch_}",
        flush=True,
    )
    evaluate_frozen(x, y, names, shared, result_dir, selection)
    generate_submission(
        project_dir, cache_dir, result_dir, x, y, ids, names, shared, helpers, selection
    )
    return selection


def run_experiment(project_dir, cache_dir, result_dir, stage="all"):
    """Run the NN experiment: stage "baseline", or "all" = "tune" + "evaluate" +
    "final" (each later stage checks that code and data did not change)."""
    project_dir = Path(project_dir).resolve()
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    shared = read_shared(project_dir)
    helpers = read_project_module(project_dir, "helpers.py", "brfss_nn_helpers")
    x, y, ids, names = load_training_data(project_dir, cache_dir)
    hashes = {
        name: file_hash(project_dir / name)
        for name in ("preprocessing.py", "validation.py", "helpers.py")
    }
    if stage == "baseline":
        return run_single_baseline(project_dir, cache_dir, result_dir)
    if stage in ("all", "tune"):
        if (result_dir / "selection.json").exists():
            raise FileExistsError(
                "A selection already exists; use a new folder for a new experiment."
            )
        manifest = dict(
            project_dir=str(project_dir),
            numpy_version=np.__version__,
            shared_hashes=hashes,
            source_hashes={
                p: file_hash(Path(__file__).parent / p) for p in ("neural_network.py",)
            },
            rows=len(y),
            positive_count=int(y.sum()),
            feature_names=names,
            train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            data_signatures={
                p: dict(
                    size=(project_dir / "data" / "dataset" / p).stat().st_size,
                    mtime_ns=(project_dir / "data" / "dataset" / p).stat().st_mtime_ns,
                )
                for p in ("x_train.csv", "y_train.csv", "x_test.csv")
            },
        )
        write_json(result_dir / "manifest.json", manifest)
        selection = tune_models(x, y, names, shared, result_dir)
    else:
        manifest = json.loads((result_dir / "manifest.json").read_text())
        if manifest["shared_hashes"] != hashes:
            raise ValueError("The shared modules changed since the tuning.")
        if manifest["train_ids_sha256"] != hashlib.sha256(ids.tobytes()).hexdigest():
            raise ValueError("The row order changed since the tuning.")
        current_signatures = {
            p: dict(
                size=(project_dir / "data" / "dataset" / p).stat().st_size,
                mtime_ns=(project_dir / "data" / "dataset" / p).stat().st_mtime_ns,
            )
            for p in ("x_train.csv", "y_train.csv", "x_test.csv")
        }
        if current_signatures != manifest["data_signatures"]:
            raise ValueError("The data files changed since the tuning.")
        current_sources = {
            p: file_hash(Path(__file__).parent / p) for p in ("neural_network.py",)
        }
        if manifest["source_hashes"] != current_sources:
            raise ValueError("The network code changed since the tuning.")
        selection = json.loads((result_dir / "selection.json").read_text())
    if stage in ("all", "evaluate"):
        evaluate_frozen(x, y, names, shared, result_dir, selection)
    if stage in ("all", "final"):
        generate_submission(
            project_dir,
            cache_dir,
            result_dir,
            x,
            y,
            ids,
            names,
            shared,
            helpers,
            selection,
        )
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("work/nn_cache"))
    parser.add_argument("--result-dir", type=Path, default=Path("nn_results"))
    parser.add_argument(
        "--stage",
        choices=["baseline", "all", "tune", "evaluate", "final"],
        default="baseline",
    )
    args = parser.parse_args()
    run_experiment(args.project_dir, args.cache_dir, args.result_dir, args.stage)
