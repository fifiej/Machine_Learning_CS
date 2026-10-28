from functools import partial
from pathlib import Path
import importlib
import numpy as np

SEED = 0

__all__ = [
    'SEED', 'to_binary', 'predict_labels', 'binary_metrics', 'confusion_matrix',
    'f1_score', 'best_f1_threshold', 'stratified_k_fold',
    'stratified_train_val_split', 'save_folds', 'load_folds',
    'prepare_brfss_pair', 'make_brfss_preprocessor', 'evaluate_holdout',
    'cross_validate', 'print_cv_summary', 'fit_predict_final',
]

def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} doit être un entier >= {minimum}.")
    return int(value)


def _vector(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in 'biuf':
        raise ValueError(f"{name} doit être un vecteur numérique 1D non vide.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contient NaN ou inf.")
    return a


def to_binary(y):
    """Copie des labels en 0/1 ; rejette notamment le mélange {-1, 0, 1}."""
    a = _vector(y, 'y')
    if not (np.isin(a, [0, 1]).all() or np.isin(a, [-1, 1]).all()):
        raise ValueError('Labels attendus : 0/1 ou -1/+1, sans mélange des deux.')
    return (a == 1).astype(np.int64)


def _scores(scores, n):
    a = _vector(scores, 'scores').astype(float, copy=False)
    if len(a) != n:
        raise ValueError('scores et labels doivent avoir la même longueur.')
    return a


def _threshold(value):
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in 'iuf' or np.isnan(a):
        raise ValueError('Le seuil doit être un scalaire numérique, sans NaN.')
    return float(a)


def predict_labels(scores, threshold=0.5, negative_label=0):
    """Probabilités : souvent seuil 0.5 ; marges : souvent seuil 0.

    negative_label=-1 fournit le codage de soumission du projet.
    Les seuils +/-inf permettent de prédire respectivement aucun/tous positifs.
    """
    a = _vector(scores, 'scores')
    if negative_label not in (0, -1):
        raise ValueError('negative_label doit valoir 0 ou -1.')
    return np.where(a >= _threshold(threshold), 1, negative_label)


def binary_metrics(y_true, y_pred):
    """TP/FP/TN/FN, accuracy, precision, recall, specificity, F1.

    Les deux vecteurs peuvent employer des codages différents. Une métrique
    dont le dénominateur est nul vaut 0. F1 est celui de la classe +1,
    pas une moyenne macro ou pondérée. Aucun arrondi avant comparaison.
    """
    y, p = to_binary(y_true), to_binary(y_pred)
    if y.shape != p.shape:
        raise ValueError('y_true et y_pred doivent avoir la même longueur.')
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fn = int(np.sum((y == 1) & (p == 0)))
    def divide(a, b):
        return float(a / b) if b else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=divide(tp + tn, len(y)),
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                specificity=divide(tn, tn + fp), f1=divide(2 * tp, 2 * tp + fp + fn))


def f1_score(y_true, y_pred):
    return binary_metrics(y_true, y_pred)['f1']


def best_f1_threshold(y_true, scores):
    """Retourne (seuil, F1 de réglage), recherche exacte en O(n log n).

    Examine toutes les décisions réalisables par score >= seuil, avec les
    scores égaux traités ensemble. En cas d'égalité, choisit le plus grand
    seuil candidat (moins de positifs). Les marges hors [0,1] sont acceptées.
    Le F1 retourné est optimisé sur ces labels : ne pas le présenter comme
    score de test indépendant. Sans positif réel, retourne (inf, 0).
    """
    y = to_binary(y_true)
    s = _scores(scores, len(y))
    order = np.argsort(-s, kind='stable')
    ranked = s[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(y) - 1]
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    # 2TP + FP + FN = nombre prédit positif + nombre réellement positif.
    values = 2.0 * tp / (ends + 1 + int(y.sum()))
    candidates = np.r_[np.inf, ranked[ends]]
    values = np.r_[0.0, values]
    k = int(np.argmax(values))
    return float(candidates[k]), float(values[k])


def stratified_k_fold(y, n_splits=5, seed=SEED):
    """Liste de (indices_train, indices_validation), déterministe.

    Chaque observation est validée exactement une fois. Tailles des plis et
    effectifs par classe diffèrent d'au plus 1. Exige les deux classes et au
    moins n_splits observations de chacune. Ne modifie pas le RNG global.
    Le même ordre de lignes, les mêmes labels, seed et n_splits sont requis
    pour partager exactement les mêmes partitions entre notebooks.
    """
    y = to_binary(y)
    k = _integer(n_splits, 'n_splits', 2)
    rng = np.random.default_rng(_integer(seed, 'seed'))
    counts = np.bincount(y, minlength=2)
    if counts.min() < k:
        raise ValueError('Chaque classe doit contenir au moins n_splits observations.')
    fold_id = np.empty(len(y), dtype=np.int64)
    offset = 0
    for c in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == c))
        fold_id[indices] = (np.arange(len(indices)) + offset) % k
        offset = (offset + len(indices)) % k
    return [(rng.permutation(np.flatnonzero(fold_id != j)),
             rng.permutation(np.flatnonzero(fold_id == j))) for j in range(k)]


def stratified_train_val_split(y, val_ratio=0.2, seed=SEED):
    """Holdout stratifié ; arrondi par classe, au moins un membre de chaque
    classe dans chaque partie. La proportion réalisée peut différer légèrement
    de val_ratio, surtout sur de très petits jeux de données.
    """
    y = to_binary(y)
    if not np.isscalar(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError('val_ratio doit être strictement entre 0 et 1.')
    rng = np.random.default_rng(_integer(seed, 'seed'))
    train, val = [], []
    for c in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == c))
        if len(indices) < 2:
            raise ValueError('Il faut au moins deux observations de chaque classe.')
        n_val = min(len(indices) - 1, max(1, int(np.floor(len(indices) * val_ratio + 0.5))))
        val.append(indices[:n_val])
        train.append(indices[n_val:])
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(val))


def _matrix(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0 or np.isinf(a).any():
        raise ValueError('x doit être une matrice 2D non vide, sans inf (NaN acceptés).')
    return a


def _validate_folds(folds, y):
    result, seen = [], np.zeros(len(y), dtype=int)
    for train, val in folds:
        pair = []
        for indices in (train, val):
            a = np.asarray(indices)
            if a.ndim != 1 or a.size == 0 or a.dtype.kind not in 'iu':
                raise ValueError('Les indices doivent être des vecteurs entiers non vides.')
            if np.any(a < 0) or np.any(a >= len(y)) or len(np.unique(a)) != len(a):
                raise ValueError('Indices hors limites ou dupliqués.')
            pair.append(a.astype(np.int64, copy=True))
        train, val = pair
        if np.intersect1d(train, val).size or len(train) + len(val) != len(y):
            raise ValueError('Chaque train doit être le complément exact de sa validation.')
        if len(np.unique(y[train])) != 2 or len(np.unique(y[val])) != 2:
            raise ValueError('Chaque partie doit contenir les deux classes.')
        seen[val] += 1
        result.append((train, val))
    if len(result) < 2 or not np.all(seen == 1):
        raise ValueError('Chaque observation doit être validée exactement une fois sur >=2 plis.')
    return result


def confusion_matrix(y_true, y_pred):
    """Matrice [[TN, FP], [FN, TP]] ; lignes = réel, colonnes = prédit."""
    m = binary_metrics(y_true, y_pred)
    return np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]], dtype=np.int64)


def _ids(sample_ids, n):
    ids = np.asarray(sample_ids)
    if ids.ndim != 1 or len(ids) != n or ids.dtype.kind not in 'iuUS':
        raise ValueError('sample_ids doit être un vecteur d’identifiants entiers ou chaînes.')
    if len(np.unique(ids)) != n:
        raise ValueError('Les identifiants doivent être uniques.')
    return ids


def save_folds(path, folds, y, sample_ids):
    """Enregistrer les partitions, leurs labels et IDs, sans écraser un fichier.

    Le fichier garde aussi l'ordre des indices dans les trains. Les IDs sont
    obligatoires pour détecter une permutation de lignes, même de même classe.
    Utiliser les train_ids renvoyés par load_csv_data. Format NumPy sans pickle.
    """
    labels = to_binary(y)
    ids = _ids(sample_ids, len(labels))
    partitions = _validate_folds(folds, labels)
    data = dict(version=np.array(1), labels=labels, sample_ids=ids,
                n_splits=np.array(len(partitions)))
    for j, (train, val) in enumerate(partitions):
        data[f'train_{j}'], data[f'val_{j}'] = train, val
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **data)


def load_folds(path, y, sample_ids):
    """Lire les partitions et vérifier les labels ET l'ordre exact des IDs."""
    labels = to_binary(y)
    ids = _ids(sample_ids, len(labels))
    with np.load(path, allow_pickle=False) as data:
        if data['version'].shape != () or int(data['version']) != 1:
            raise ValueError('Version de fichier de partitions non prise en charge.')
        if not np.array_equal(data['labels'], labels) or not np.array_equal(data['sample_ids'], ids):
            raise ValueError('Labels ou ordre des IDs différents des partitions enregistrées.')
        k = _integer(data['n_splits'].item(), 'n_splits', 2)
        partitions = [(data[f'train_{j}'], data[f'val_{j}']) for j in range(k)]
    return _validate_folds(partitions, labels)


def _names(names, n):
    names = list(names)
    if len(names) != n or not all(isinstance(v, str) and v for v in names):
        raise ValueError('names doit contenir un nom non vide par colonne, sans Id.')
    if len(set(names)) != len(names):
        raise ValueError('Les noms de colonnes doivent être uniques.')
    return names


def prepare_brfss_pair(x_train, x_eval, *, names, clean_data=True,
                       categorical_columns=(), add_missing_indicators=False,
                       min_nan=0.05, max_nan=0.9,
                       interaction_pairs=(), square_columns=(),
                       standardize=True, add_bias=False,
                       preprocessing_module=None, return_names=False):
    """Appliquer le preprocessing.py partagé à une paire train/évaluation.

    Toutes les décisions apprises utilisent x_train uniquement :
      1. compute_groups -> clean sur les deux parties (si clean_data=True).
      2. fit_column_filter sur le train, même masque sur l'évaluation.
      3. add_interactions sur les variables originales nettoyées.
      4. nan_indicators AVANT one_hot et imputation.
      5. one_hot : catégories déterminées sur le train seulement.
      6. Retrait des colonnes entièrement NaN sur le train, puis
         fit_preprocessing et apply_preprocessing partagés (imputation MOYENNE).

    clean_data=False : pas de nettoyage BRFSS ni de filtre à 90% ; seuls les
    NaN sont imputés et les colonnes entièrement manquantes retirées. Les
    expansions restent commandées par leurs options explicites.
    standardize=False : imputation avec la même moyenne, sans centrer/réduire.
    add_bias=False convient à un réseau ayant ses propres biais, et aux arbres.
    add_bias=True convient aux fonctions linéaires sans intercept intégré.

    Les interactions précèdent le one-hot ; elles utilisent les noms originaux.
    Une colonne demandée qui n'existe pas dans names provoque une erreur.
    Si le nettoyage la supprime dans ce train, son expansion est ignorée comme
    dans le module partagé. Le nombre de features peut varier entre plis.

    NaN et catégories inconnues donnent des dummies toutes à zéro, conformément
    au one_hot partagé. Les indicateurs ne sont ajoutés que selon les ratios de
    NaN DU TRAIN. Les règles métier du module partagé sont reprises telles
    quelles, sans certification de leur justesse sémantique.

    preprocessing_module=None importe 'preprocessing' au moment de l'appel.
    Retour : (train_transformé, eval_transformé), ou ces deux matrices et les
    noms finaux si return_names=True. Aucune entrée n'est modifiée.
    """
    a, b = _matrix(x_train).copy(), _matrix(x_eval).copy()
    if a.shape[1] != b.shape[1]:
        raise ValueError('Train et évaluation doivent avoir les mêmes colonnes initiales.')
    names = _names(names, a.shape[1])
    categories, squares = list(categorical_columns), list(square_columns)
    pairs = list(interaction_pairs)
    if any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in pairs):
        raise ValueError('Chaque interaction doit être une paire de noms.')
    requested = categories + squares + [n for pair in pairs for n in pair]
    if any(not isinstance(n, str) or n not in names for n in requested):
        raise ValueError('Une expansion demandée référence une colonne initiale inconnue.')
    if len(set(categories)) != len(categories) or len(set(squares)) != len(squares):
        raise ValueError('Colonnes dupliquées dans les expansions demandées.')
    pair_keys = [tuple(sorted(p)) for p in pairs]
    if len(set(pair_keys)) != len(pair_keys) or any(a == b and a in squares for a, b in pairs):
        raise ValueError('Interactions dupliquées (y compris carré et produit de soi-même).')
    if not (0 <= min_nan <= max_nan <= 1):
        raise ValueError('Il faut 0 <= min_nan <= max_nan <= 1.')
    module = preprocessing_module
    if module is None:
        module = importlib.import_module('preprocessing')
    required = ['fit_preprocessing', 'apply_preprocessing']
    if clean_data:
        required += ['compute_groups', 'clean', 'fit_column_filter']
    if categories:
        required.append('one_hot')
    if add_missing_indicators:
        required.append('nan_indicators')
    if pairs or squares:
        required.append('add_interactions')
    absent = [n for n in required if not callable(getattr(module, n, None))]
    if absent:
        raise ValueError('Mettre à jour preprocessing.py ; fonctions absentes : ' + ', '.join(absent))
    if clean_data:
        groups = module.compute_groups(a, names)
        a, cleaned_names = module.clean(a, names, groups)
        b, eval_names = module.clean(b, names, groups)
        if cleaned_names != eval_names:
            raise ValueError('Le nettoyage a produit des colonnes différentes entre train et évaluation.')
        names = cleaned_names
        if a.shape[1]:
            keep = module.fit_column_filter(a)
            a, b = a[:, keep], b[:, keep]
            names = [n for n, selected in zip(names, keep) if selected]
    if a.shape[1] == 0:
        raise ValueError('Le nettoyage a supprimé toutes les colonnes de ce train.')
    if pairs or squares:
        a, b, names = module.add_interactions(a, b, names, pairs, squares)
    if add_missing_indicators:
        a, b, names = module.nan_indicators(a, b, names, min_nan=min_nan, max_nan=max_nan)
    if categories:
        a, b, names = module.one_hot(a, b, names, categories)
    # Un produit peut être entièrement NaN, même si ses facteurs ne le sont pas.
    # Le module partagé utilise nanmean : retirer ces colonnes avant son fit.
    keep = np.any(~np.isnan(a), axis=0)
    a, b = a[:, keep], b[:, keep]
    names = [n for n, selected in zip(names, keep) if selected]
    if not names:
        raise ValueError('Aucune colonne avec une valeur observée dans ce train.')
    _names(names, a.shape[1])
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError('Débordement numérique dans les interactions ou les données.')
    mean, std = module.fit_preprocessing(a)
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (a.shape[1],) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Le prétraitement a produit des moyennes/écarts-types invalides.')
    if standardize:
        a, b = module.apply_preprocessing(a, mean, std), module.apply_preprocessing(b, mean, std)
        # Contrat de la version fournie : colonne de 1 ajoutée en première position.
        if a.shape[1] != len(names)+1 or b.shape[1] != len(names)+1 or not np.all(a[:, 0] == 1) or not np.all(b[:, 0] == 1):
            raise ValueError('apply_preprocessing doit ajouter une colonne de biais en tête.')
        if not add_bias:
            a, b = a[:, 1:], b[:, 1:]
    else:
        a, b = np.where(np.isnan(a), mean, a), np.where(np.isnan(b), mean, b)
        if add_bias:
            a, b = np.c_[np.ones(len(a)), a], np.c_[np.ones(len(b)), b]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('NaN/inf restant après prétraitement.')
    final_names = (['__bias__'] if add_bias else []) + names
    return (a, b, final_names) if return_names else (a, b)


def make_brfss_preprocessor(names, **options):
    """Créer preprocess(train, eval), à passer aux fonctions d'évaluation.

    Toutes les statistiques sont recalculées uniquement sur le train de CHAQUE
    appel. Aucun état ajusté n'est partagé entre les plis. Exemple :
        preprocess = make_brfss_preprocessor(names, add_bias=False)
    """
    if options.get('return_names', False):
        raise ValueError('Le callback de validation doit retourner deux matrices, sans noms.')
    options = options.copy()
    for key in ('categorical_columns', 'square_columns'):
        if key in options:
            options[key] = tuple(options[key])
    if 'interaction_pairs' in options:
        options['interaction_pairs'] = tuple(tuple(p) for p in options['interaction_pairs'])
    return partial(prepare_brfss_pair, names=tuple(names), **options)


def _validate_xy(x, y):
    x, y = _matrix(x), to_binary(y)
    if len(x) != len(y):
        raise ValueError('x et y doivent avoir le même nombre de lignes.')
    return x, y


def _threshold_mode(threshold):
    if isinstance(threshold, str) and threshold == 'inner':
        return threshold
    return _threshold(threshold)


def _run_model(x_train, y_train, x_eval, fit_predict, preprocess, seed):
    a, b = x_train.copy(), x_eval.copy()
    if preprocess is not None:
        a, b = preprocess(a, b)
    a, b = _matrix(a), _matrix(b)
    if len(a) != len(y_train) or len(b) != len(x_eval) or a.shape[1] != b.shape[1]:
        raise ValueError('Le prétraitement a changé les lignes ou désaligné les colonnes.')
    # NaN possibles uniquement si un modèle sait explicitement les traiter.
    return _scores(fit_predict(a, y_train.copy(), b, seed), len(b)).copy()


def _evaluate_partition(x, y, train, val, fit_predict, preprocess,
                        threshold, inner_val_ratio, seeds):
    chosen, tuning_f1 = threshold, None
    if isinstance(threshold, str):
        it, iv = stratified_train_val_split(y[train], inner_val_ratio, seeds[0])
        inner_scores = _run_model(x[train[it]], y[train[it]], x[train[iv]],
                                  fit_predict, preprocess, seeds[1])
        chosen, tuning_f1 = best_f1_threshold(y[train[iv]], inner_scores)
    scores = _run_model(x[train], y[train], x[val], fit_predict, preprocess, seeds[2])
    pred = predict_labels(scores, chosen)
    return dict(n_train=len(train), n_val=len(val), threshold=float(chosen),
                inner_tuning_f1=tuning_f1, metrics=binary_metrics(y[val], pred),
                scores=scores, predictions=pred)


def _seeds(seed, count):
    sequence = np.random.SeedSequence(_integer(seed, 'seed'))
    return [int(c.generate_state(1)[0]) for c in sequence.spawn(count)]


def evaluate_holdout(x, y, fit_predict, *, split=None, val_ratio=0.2,
                     seed=SEED, preprocess=None, threshold=0.5,
                     inner_val_ratio=0.2):
    """Essai rapide sur un seul train/validation stratifié.

    split=(indices_train, indices_val), par exemple folds[0], permet de partager
    exactement le même holdout. Sinon il est créé avec val_ratio et seed.
    Chaque partie doit être le complément de l'autre et avoir les deux classes.
    threshold='inner' : deux entraînements et réglage interne du seuil.
    Le résultat contient métriques, scores, prédictions et indices originaux.
    """
    x, y = _validate_xy(x, y)
    chosen_mode = _threshold_mode(threshold)
    if split is None:
        split = stratified_train_val_split(y, val_ratio, seed)
    # Les deux plis complémentaires permettent de réutiliser la vérification.
    train, val = _validate_folds([split, (split[1], split[0])], y)[0]
    result = _evaluate_partition(x, y, train, val, fit_predict, preprocess,
                                 chosen_mode, inner_val_ratio, _seeds(seed, 3))
    result.update(train_indices=train, val_indices=val)
    return result


def cross_validate(x, y, fit_predict, *, n_splits=5, seed=SEED,
                   folds=None, preprocess=None, threshold=0.5,
                   inner_val_ratio=0.2, verbose=False):
    """Validation croisée ; prétraitement réajusté dans chaque train.

    fit_predict(x_train, y_train_01, x_eval, seed) -> scores 1D, modèle neuf.
    preprocess(train_raw, eval_raw) -> deux matrices ; None transmet le brut.
    threshold : scalaire fixé à l'avance, ou 'inner' pour un réglage interne.
    folds : partitions communes facultatives, intégralement vérifiées.

    Retour : résultats par pli, moyenne/écart-type descriptif (ddof=0), scores
    et prédictions out-of-fold dans l'ordre ORIGINAL, métriques OOF agrégées.
    F1 moyen et F1 agrégé sont distincts. L'écart-type n'est pas un intervalle
    de confiance. Les scores de modèles différents ne sont pas nécessairement
    calibrés pareil : les prédictions OOF utilisent le seuil de leur propre pli.

    threshold='inner' coûte 2 entraînements par pli ; le modèle final de chaque
    pli utilise tout le train externe. Le changement de calibration après ce
    réentraînement peut dégrader le seuil choisi : le score externe mesure
    honnêtement cette procédure. La fonction ne choisit pas un modèle gagnant.
    """
    x, y = _validate_xy(x, y)
    mode = _threshold_mode(threshold)
    partitions = _validate_folds(
        stratified_k_fold(y, n_splits, seed) if folds is None else folds, y)
    seeds = _seeds(seed, 3*len(partitions))
    oof_scores, oof_pred = np.empty(len(y)), np.empty(len(y), dtype=int)
    oof_fold = np.empty(len(y), dtype=int)
    results = []
    for j, (train, val) in enumerate(partitions):
        result = _evaluate_partition(x, y, train, val, fit_predict, preprocess,
                                     mode, inner_val_ratio, seeds[3*j:3*j+3])
        oof_scores[val] = result.pop('scores')
        oof_pred[val] = result.pop('predictions')
        oof_fold[val] = j
        result['fold'] = j
        results.append(result)
        if verbose:
            print(f"Pli {j+1}/{len(partitions)} : F1={result['metrics']['f1']:.4f}, "
                  f"seuil={result['threshold']:.6g}", flush=True)
    keys = ('accuracy', 'precision', 'recall', 'specificity', 'f1')
    mean = {k: float(np.mean([r['metrics'][k] for r in results])) for k in keys}
    std = {k: float(np.std([r['metrics'][k] for r in results])) for k in keys}
    return dict(folds=results, mean=mean, std=std, oof_scores=oof_scores,
                oof_predictions=oof_pred, oof_fold=oof_fold,
                oof_metrics=binary_metrics(y, oof_pred))


def print_cv_summary(result):
    """Afficher les métriques de cross_validate sans arrondir les données stockées."""
    print('Pli   F1       Précision  Rappel    Accuracy  Seuil')
    for r in result['folds']:
        m = r['metrics']
        print(f"{r['fold']+1:>3}   {m['f1']:.4f}   {m['precision']:.4f}     "
              f"{m['recall']:.4f}    {m['accuracy']:.4f}    {r['threshold']:.6g}")
    print(f"F1 moyen : {result['mean']['f1']:.4f} ; écart-type : {result['std']['f1']:.4f}")
    print(f"F1 OOF agrégé : {result['oof_metrics']['f1']:.4f}")


def fit_predict_final(x_train, y_train, x_test, fit_predict, *, preprocess=None,
                      threshold=0.5, seed=SEED, inner_val_ratio=0.2,
                      negative_label=-1):
    """Réentraîner sur tout le train et prédire le test dans son ordre original.

    Seuil numérique fixé à partir du développement, ou 'inner' pour sélectionner
    le seuil sur un holdout interne avant réentraînement complet. Le F1 interne
    renvoyé est un score de réglage, jamais un F1 du test. Aucun label de test
    n'est nécessaire. Aucun CSV ni envoi réseau n'est effectué.

    Retour : scores, predictions (-1/+1 par défaut), threshold, inner_tuning_f1.
    Le modèle n'est pas conservé : l'interface fit_predict ne le retourne pas.
    """
    x_train, y = _validate_xy(x_train, y_train)
    x_test = _matrix(x_test)
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError('Le train et le test doivent avoir les mêmes colonnes initiales.')
    if negative_label not in (0, -1):
        raise ValueError('negative_label doit valoir 0 ou -1.')
    chosen, tuning_f1 = _threshold_mode(threshold), None
    seeds = _seeds(seed, 3)
    if isinstance(chosen, str):
        it, iv = stratified_train_val_split(y, inner_val_ratio, seeds[0])
        s = _run_model(x_train[it], y[it], x_train[iv], fit_predict, preprocess, seeds[1])
        chosen, tuning_f1 = best_f1_threshold(y[iv], s)
    scores = _run_model(x_train, y, x_test, fit_predict, preprocess, seeds[2])
    return dict(scores=scores, predictions=predict_labels(scores, chosen, negative_label),
                threshold=float(chosen), inner_tuning_f1=tuning_f1)
