# BRFSS Heart-Disease Classification — EPFL Machine Learning Project 1

This project predicts the positive heart-disease label from health and lifestyle responses in the 2015 Behavioral Risk Factor Surveillance System (BRFSS). It documents the data pipeline, the six required course methods, additional NumPy models, cross-validation, and the CSV submission workflow.

## Project files

| File or folder | Purpose |
|---|---|
| `implementations.py` | The six required linear and logistic methods. |
| `preprocessing.py` | Shared BRFSS cleaning, feature engineering, imputation, scaling, and baseline helpers. |
| `validation.py` | Binary metrics, F1 threshold selection, stratified splits, cross-validation, and final prediction helpers. |
| `run.py` | Reproduces the final regularized-logistic submission pipeline. |
| `trees.py` | Decision tree, random forest, and logistic-loss gradient boosting. |
| `neural_network.py` | NumPy MLP and its tuning, holdout-evaluation, and submission workflow. |
| `exploration.ipynb` | Data exploration and preprocessing experiments. |
| `linear_models.ipynb` | Comparison of the six required methods. |
| `nonlinear_models.ipynb` | Decision trees, forests, boosting, ensembles, and neural-network experiments. |
| `cv_final.ipynb` | Five-fold cross-validation for the final logistic pipeline. |
| `data/dataset_to_release.zip` | Provided dataset archive. Extracted data is local and ignored by Git. |
| `report/report.pdf` | Project report; its LaTeX source is in `report/report.tex`. |
| `figures/` | Figures and selected experiment summaries used by the report. |

## Setup and data

Use the course Python environment, or install the packages needed for the scripts and notebooks:

```bash
python -m pip install numpy matplotlib jupyter
```

From the repository root, extract the provided dataset:

```bash
unzip data/dataset_to_release.zip -d data
```

This creates `data/dataset/`, which must contain `x_train.csv`, `y_train.csv`, `x_test.csv`, and `sample_submission.csv`. Keep the supplied filenames and IDs unchanged. The extracted data should not be committed.

## Data preparation and features

The shared pipeline in `preprocessing.py` learns cleaning and feature rules from the training data, then applies them to validation or test data:

1. BRFSS columns are grouped by their missingness, value range, and survey-code patterns. Special answers such as “don't know” and “refused” are converted to missing values where appropriate; yes/no answers encoded as `1/2` become `1/0`, with column-specific handling for fields such as `DIABETE3` and `BPHIGH4`.
2. Identifier and administrative fields, survey weights, columns with more than 90% missing values, and constant columns are removed.
3. The logistic pipeline one-hot encodes `MARITAL`, `EMPLOY1`, `RENTHOM1`, `PERSDOC2`, `_RACE`, `SXORIENT`, and `TRNSGNDR`.
4. It adds six pairwise interactions (age with high blood pressure, diabetes, general health, sex, and stroke; and general health with difficulty walking) and the squares of age and BMI. Exact duplicate features are then removed.
5. Missing values are imputed with training means. Features are standardized with training statistics, and a bias column is added for the linear models.

The neural network uses the shared cleaning rules, one-hot encodes `SEX`, `MARITAL`, and `EMPLOY1`, adds missing-value indicators, imputes with training means, and standardizes without a separate bias column. Tree models use mean imputation and training-quantile bins instead of standardization. Cross-validation fits preprocessing independently inside each fold to avoid validation leakage.

## Models

### Six required methods

`implementations.py` provides:

- `mean_squared_error_gd` and `mean_squared_error_sgd`
- `least_squares` and `ridge_regression`
- `logistic_regression` and `reg_logistic_regression`

The additional methods are implemented without external machine-learning estimators. `trees.py` uses quantile bins (up to 32 per feature), Gini-based decision trees, bootstrap random forests, and gradient boosting that fits shallow trees to logistic-loss gradients and Hessians. `neural_network.py` implements a one-hidden-layer ReLU MLP with a sigmoid output, weighted binary cross-entropy, L2 regularization, and Adam. Its hidden size, learning rate, regularization, positive-class weight, batch size, and epoch limit can be tuned in code.

## Validation and model selection

`validation.py` supplies deterministic stratified folds (seed `0`), binary metrics, confusion matrices, exact F1 threshold search, and `cross_validate`. The `cv_final.ipynb` notebook runs five-fold cross-validation. Each fold learns preprocessing and selects its threshold on training-side data; the held-out fold is used only for evaluation.

The final `run.py` pipeline uses a fixed 80/20 train/validation split to choose a decision threshold, then retrains regularized logistic regression on all labeled rows before predicting the test set. The neural-network experiment separates development data, tuning data, and a holdout: it freezes the model choice and threshold before evaluating the holdout.

## Reported results

| Model / evaluation | F1 | Accuracy |
|---|---:|---:|
| Regularized logistic regression, 5-fold CV | 0.4268 ± 0.0035 | 0.8710 ± 0.0069 |
| Logistic regression + gradient boosting, 5-fold CV | 0.4314 ± 0.0040 | — |
| Final regularized-logistic AIcrowd submission | 0.441 | 0.878 |
| Neural-network local holdout | 0.4171 | 0.8783 |

The report identifies regularized logistic regression as the final AIcrowd model. The neural-network holdout is a separate split; its score should not be compared as if it were the same evaluation set. `figures/nn/` contains the saved NN selection, tuning, manifest, and holdout metric files.

## Reproduce a submission

From the repository root, run the final logistic pipeline:

```bash
python run.py
```

It writes `submission.csv` in AIcrowd's expected `Id,Prediction` format, with labels `-1` and `+1`. The script prints its validation F1 and chosen threshold before fitting on all labeled data.

To run the simple NumPy neural-network baseline instead:

```bash
python neural_network.py --stage baseline --project-dir . --cache-dir work/nn_cache --result-dir nn_results_baseline
```

Use a new result directory for each run. For the full tuning, frozen holdout evaluation, and final prediction workflow, use `--stage all` and a fresh result directory. Neural-network CSVs are saved inside that result directory; the script does not upload them to AIcrowd.

## Reproducibility notes

The shared split seed is `0`. The final logistic configuration is set in `run.py`; feature lists and transformations are in `preprocessing.py`. The neural-network experiment saves its configuration, threshold, data signatures, and evaluation metrics in its result folder. Generated submissions, extracted data, and `work/` caches are local files and are ignored by Git.
