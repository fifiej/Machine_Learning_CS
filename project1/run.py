"""
run.py: reproduces our final AIcrowd submission.

Pipeline:
    BRFSS cleaning -> one-hot encoding -> interactions
    -> regularized logistic regression (implementations.py)
    -> decision threshold chosen on a validation split (max F1)
    -> retrain on all the training data -> submission.csv

Usage:
    python run.py
    (x_train.csv, y_train.csv and x_test.csv must be in data/dataset/)
"""
import csv
import os

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import reg_logistic_regression
from preprocessing import (compute_groups, clean, fit_column_filter, one_hot,
                           add_interactions, split_train_val, fit_preprocessing,
                           apply_preprocessing, sigmoid, best_threshold)

# ---------------------------------------------------------------------
# Settings (all chosen with the experiments in the notebooks)
# ---------------------------------------------------------------------
DATA_PATH = "data/dataset"
OUTPUT = "submission.csv"

# regularized logistic regression (best lambda in linear_models.ipynb)
LAMBDA, GAMMA, MAX_ITERS = 1e-4, 0.5, 1000

# categorical columns without a natural order -> one-hot encoding
CATEGORICAL = ["MARITAL", "EMPLOY1", "RENTHOM1", "PERSDOC2",
               "_RACE", "SXORIENT", "TRNSGNDR"]

# products of two features and squares (see baseline.ipynb)
INTERACTIONS = [
    ("_AGE80", "BPHIGH4"),    # age x high blood pressure
    ("_AGE80", "DIABETE3"),   # age x diabetes
    ("_AGE80", "GENHLTH"),    # age x general health
    ("_AGE80", "SEX"),        # age x sex
    ("_AGE80", "CVDSTRK3"),   # age x stroke
    ("GENHLTH", "DIFFWALK"),  # general health x difficulty walking
]
SQUARES = ["_AGE80", "_BMI5"]


def load_data():
    """Load the CSV files and the column names.
    Returns x_train, x_test, y01 (labels 0/1), test_ids, names."""
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    with open(os.path.join(DATA_PATH, "x_train.csv")) as f:
        names = next(csv.reader(f))[1:]          # column names without "Id"
    y01 = np.where(y_train == 1, 1, 0).astype(float)   # -1/1 -> 0/1
    return x_train, x_test, y01, test_ids, names


def build_features(x_train, x_test, names):
    """Cleaning + column filter + one-hot + interactions.
    Every rule is learned on x_train only, then applied identically to x_test."""
    groups = compute_groups(x_train, names)
    x_tr, kept = clean(x_train, names, groups)
    x_te, _ = clean(x_test, names, groups)

    keep = fit_column_filter(x_tr)                # drop >90 % NaN and constant columns
    x_tr, x_te = x_tr[:, keep], x_te[:, keep]
    kept = [n for n, k in zip(kept, keep) if k]

    x_tr, x_te, kept = one_hot(x_tr, x_te, kept, CATEGORICAL)
    x_tr, x_te, kept = add_interactions(x_tr, x_te, kept, INTERACTIONS, SQUARES)
    return x_tr, x_te


def train(x, y):
    """Fill NaNs, standardize and train the regularized logistic regression.
    Returns w, mean, std (needed to transform other data the same way)."""
    mean, std = fit_preprocessing(x)
    tx = apply_preprocessing(x, mean, std)        # with bias column
    initial_w = np.zeros(tx.shape[1])             # new array at every call
    w, _ = reg_logistic_regression(y, tx, LAMBDA, initial_w, MAX_ITERS, GAMMA)
    return w, mean, std


def predict_proba(x, w, mean, std):
    """Predicted probability of heart disease for each row of x."""
    return sigmoid(apply_preprocessing(x, mean, std) @ w)


def main():
    np.random.seed(0)   # reproducibility

    x_train, x_test, y01, test_ids, names = load_data()
    x_train, x_test = build_features(x_train, x_test, names)
    print("Features:", x_train.shape[1])

    # 1. choose the decision threshold on a validation split (80 / 20)
    i_tr, i_val = split_train_val(len(x_train))
    w, mean, std = train(x_train[i_tr], y01[i_tr])
    t, f1_val = best_threshold(y01[i_val], predict_proba(x_train[i_val], w, mean, std))
    print(f"Validation F1 = {f1_val:.3f} with threshold {t:.2f}")

    # 2. retrain on all the training data and predict the test set
    w, mean, std = train(x_train, y01)
    y_pred = np.where(predict_proba(x_test, w, mean, std) >= t, 1, -1)   # labels -1/1

    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(f"{OUTPUT} written: {len(y_pred)} rows, "
          f"{np.mean(y_pred == 1):.1%} predicted positive")


if __name__ == "__main__":
    main()