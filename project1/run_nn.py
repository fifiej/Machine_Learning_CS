"""Expérience NN reproductible, NumPy uniquement.

CLI : python run_nn.py --project-dir . --cache-dir work/nn_cache --result-dir nn_results
Le CSV est créé localement ; aucun envoi AIcrowd n'est effectué.
Le jeu externe (folds[0]) n'intervient ni dans le choix d'architecture, ni dans
l'early stopping, ni dans le choix de seuil/ensemble. Il est évalué une fois.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from nn_model import BinaryMLP
from validation import (to_binary, stratified_k_fold, stratified_train_val_split,
                        prepare_brfss_pair, best_f1_threshold, binary_metrics)

SEED=0
PREPROCESS_OPTIONS=dict(categorical_columns=['SEX','MARITAL','EMPLOY1'],
                       add_missing_indicators=True, standardize=True, add_bias=False)
CANDIDATES=[
    dict(name='linear_adam',hidden_size=0,learning_rate=.003,l2=.001,positive_weight=1.,epochs=25),
    dict(name='nn32',hidden_size=32,learning_rate=.001,l2=.001,positive_weight=1.,epochs=30),
    dict(name='nn64',hidden_size=64,learning_rate=.001,l2=.003,positive_weight=1.,epochs=30),
    dict(name='nn32_weighted',hidden_size=32,learning_rate=.001,l2=.003,positive_weight=3.,epochs=30),
    dict(name='nn64_low_l2',hidden_size=64,learning_rate=.001,l2=.0001,positive_weight=1.,epochs=35),
    dict(name='nn128_low_l2',hidden_size=128,learning_rate=.0005,l2=.0001,positive_weight=1.,epochs=35),
    dict(name='nn64_weighted',hidden_size=64,learning_rate=.001,l2=.0003,positive_weight=3.,epochs=35),
]


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def read_project_module(project_dir, filename, module_name):
    path=Path(project_dir)/filename
    spec=importlib.util.spec_from_file_location(module_name,path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Impossible de charger {path}.')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_shared(project_dir):
    return read_project_module(project_dir,'preprocessing.py','brfss_nn_preprocessing')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv_features(path,cache_dir):
    """Lire les CSV par lignes pour éviter la grosse mémoire temporaire de genfromtxt.

    Cache .npy local, invalidé si chemin, taille ou mtime du CSV changent.
    Il ne s'agit pas de données externes : uniquement des copies des CSV fournis.
    """
    path=Path(path).resolve()
    cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    prefix=cache_dir/path.stem
    meta_path=prefix.with_suffix('.json')
    feature_path=prefix.with_suffix('.npy')
    ids_path=cache_dir/(path.stem+'_ids.npy')
    stat=path.stat()
    signature=dict(source=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if meta_path.exists() and feature_path.exists() and ids_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']==signature:
            return np.load(feature_path,mmap_mode='r'),np.load(ids_path),meta['names']
    print(f'Lecture et cache de {path.name}...',flush=True)
    with path.open(newline='') as f:
        reader=csv.reader(f)
        header=next(reader)
        n=sum(1 for _ in reader)
    if header[0]!='Id':
        raise ValueError('Première colonne attendue : Id.')
    x=np.lib.format.open_memmap(feature_path,mode='w+',dtype='float64',shape=(n,len(header)-1))
    ids=np.empty(n,dtype=np.int64)
    with path.open(newline='') as f:
        reader=csv.reader(f)
        next(reader)
        for i,row in enumerate(reader):
            if len(row)!=len(header):
                raise ValueError(f'Ligne CSV mal formée : {i+2}')
            ids[i]=int(row[0])
            x[i]=np.asarray([value if value else 'nan' for value in row[1:]],dtype=float)
    x.flush()
    del x
    if len(np.unique(ids))!=len(ids):
        raise ValueError('IDs dupliqués.')
    np.save(ids_path,ids)
    write_json(meta_path,dict(signature=signature,names=header[1:]))
    return np.load(feature_path,mmap_mode='r'),ids,header[1:]


def load_training_data(project_dir,cache_dir):
    data=Path(project_dir)/'data'/'dataset'
    x,ids,names=_csv_features(data/'x_train.csv',cache_dir)
    targets=np.loadtxt(data/'y_train.csv',delimiter=',',skiprows=1,dtype=np.int64)
    if targets.shape!=(len(x),2) or not np.array_equal(ids,targets[:,0]):
        raise ValueError('Les IDs des features et des labels ne sont pas alignés.')
    return x,to_binary(targets[:,1]),ids,names


def prepare(xa,xb,names,shared):
    a,b=prepare_brfss_pair(xa,xb,names=names,preprocessing_module=shared,**PREPROCESS_OPTIONS)
    # Float32 pour les multiplications du réseau ; le fit partagé reste en float64.
    return a.astype(np.float32),b.astype(np.float32)


def tune_models(x,y,names,shared,result_dir,candidates=None):
    result_dir=Path(result_dir)
    result_dir.mkdir(parents=True,exist_ok=True)
    development,holdout=stratified_k_fold(y,seed=SEED)[0]
    local_train,local_tune=stratified_train_val_split(y[development],val_ratio=.2,seed=SEED+1)
    train,tuning=development[local_train],development[local_tune]
    np.savez_compressed(result_dir/'experiment_splits.npz',train=train,tuning=tuning,
                        development=development,holdout=holdout)
    print(f'Train={len(train)}, réglage={len(tuning)}, holdout réservé={len(holdout)}',flush=True)
    a,b=prepare(x[train],x[tuning],names,shared)
    print(f'Features réseau : {a.shape[1]}',flush=True)
    models,scores,rows=[],[],[]
    def train_candidate(config,seed,label):
        params={k:v for k,v in config.items() if k!='name'}
        model=BinaryMLP(**params,seed=seed)
        def monitor(m):
            s=m.predict_scores(b)
            threshold,f1=best_f1_threshold(y[tuning],s)
            metrics=binary_metrics(y[tuning],(s>=threshold).astype(int))
            return dict(selection_score=f1,threshold=threshold,accuracy=metrics['accuracy'])
        start=time.monotonic()
        print(f'\nEssai {label}',flush=True)
        model.fit(a,y[train],monitor=monitor,patience=5,verbose=True)
        s=model.predict_scores(b)
        threshold,f1=best_f1_threshold(y[tuning],s)
        row=dict(name=label,config=model.config,best_epoch=model.best_epoch_,
                 threshold=threshold,tuning=binary_metrics(y[tuning],(s>=threshold).astype(int)),
                 tuning_at_05=binary_metrics(y[tuning],(s>=.5).astype(int)),
                 seconds=time.monotonic()-start,checkpoint=f'{label}.npz')
        model.save(result_dir/row['checkpoint'])
        write_json(result_dir/f'{label}_history.json',model.history_)
        models.append(model); scores.append(s); rows.append(row)
        write_json(result_dir/'tuning_trials.json',rows)
        return row
    for candidate in (CANDIDATES if candidates is None else candidates):
        train_candidate(candidate,SEED,candidate['name'])
    # Optimiser le rôle réseau : sélectionner un réseau, garder la logistique comme référence.
    neural=[i for i,r in enumerate(rows) if r['config']['hidden_size']>0]
    if not neural:
        raise ValueError('Au moins un candidat réseau est requis.')
    best=max(neural,key=lambda i:rows[i]['tuning']['f1'])
    config=rows[best]['config'].copy()
    config.pop('seed')
    selected=[best]
    for seed in (1,2):
        train_candidate(config,seed,f"{rows[best]['name']}_seed{seed}")
        selected.append(len(rows)-1)
    ensemble_scores=np.mean([scores[i] for i in selected],axis=0)
    ensemble_t,ensemble_f1=best_f1_threshold(y[tuning],ensemble_scores)
    best_single=max(neural+[len(rows)-2,len(rows)-1],key=lambda i:rows[i]['tuning']['f1'])
    # Toutes ces décisions utilisent uniquement le jeu de réglage, pas le holdout.
    if ensemble_f1>rows[best_single]['tuning']['f1']:
        choice=selected
        threshold=ensemble_t
        tuning_metrics=binary_metrics(y[tuning],(ensemble_scores>=threshold).astype(int))
        kind='ensemble_three_seeds'
    else:
        choice=[best_single]
        threshold=rows[best_single]['threshold']
        tuning_metrics=rows[best_single]['tuning']
        kind='single_network'
    members=[]
    for i in choice:
        config=rows[i]['config'].copy()
        config['epochs']=rows[i]['best_epoch']
        members.append(dict(name=rows[i]['name'],config=config))
    selection=dict(kind=kind,members=members,threshold=threshold,
                   tuning_metrics=tuning_metrics,preprocess_options=PREPROCESS_OPTIONS,
                   selection_rule='maximum tuning F1; accuracy reported, not a selection criterion',
                   outer_seed=SEED,inner_seed=SEED+1,
                   sample_counts=dict(train=len(train),tuning=len(tuning),holdout=len(holdout)),
                   ensemble_tuning_f1=ensemble_f1)
    # Cette décision est écrite avant toute mesure sur le holdout.
    write_json(result_dir/'selection.json',selection)
    print(f"\nSélection figée : {kind}, F1 de réglage={tuning_metrics['f1']:.4f}, seuil={threshold:.6f}",flush=True)
    return selection


def train_fixed_members(a,y,b,selection,result_dir,prefix):
    aggregate=np.zeros(len(b),dtype=float)
    for member in selection['members']:
        print(f"Réentraînement {prefix}: {member['name']}, {member['config']['epochs']} époques",flush=True)
        model=BinaryMLP(**member['config']).fit(a,y,verbose=True)
        model.save(Path(result_dir)/f"{prefix}_{member['name']}.npz")
        aggregate+=model.predict_scores(b)/len(selection['members'])
    return aggregate


def evaluate_frozen(x,y,names,shared,result_dir,selection):
    result_dir=Path(result_dir)
    if (result_dir/'holdout_metrics.json').exists():
        raise FileExistsError('Holdout déjà évalué dans ce dossier ; lire le résultat existant.')
    splits=np.load(result_dir/'experiment_splits.npz',allow_pickle=False)
    development,holdout=splits['development'],splits['holdout']
    a,b=prepare(x[development],x[holdout],names,shared)
    scores=train_fixed_members(a,y[development],b,selection,result_dir,'holdout')
    pred=(scores>=selection['threshold']).astype(int)
    metrics=dict(selected_threshold=binary_metrics(y[holdout],pred),
                 at_05=binary_metrics(y[holdout],(scores>=.5).astype(int)),
                 always_negative=binary_metrics(y[holdout],np.zeros(len(holdout),dtype=int)),
                 threshold=selection['threshold'],n=len(holdout),
                 note='Configuration, époques, ensemble et seuil figés avant cette évaluation.')
    np.savez_compressed(result_dir/'holdout_predictions.npz',indices=holdout,y=y[holdout],scores=scores,predictions=pred)
    write_json(result_dir/'holdout_metrics.json',metrics)
    print('\nÉVALUATION HOLDOUT',json.dumps(metrics,indent=2),flush=True)
    return metrics


def generate_submission(project_dir,cache_dir,result_dir,x,y,ids,names,shared,helpers,selection):
    result_dir=Path(result_dir)
    xt,test_ids,test_names=_csv_features(Path(project_dir)/'data'/'dataset'/'x_test.csv',cache_dir)
    if test_names!=names:
        raise ValueError('Colonnes train/test différentes.')
    a,b=prepare(x,xt,names,shared)
    scores=train_fixed_members(a,y,b,selection,result_dir,'final')
    pred=np.where(scores>=selection['threshold'],1,-1)
    if len(pred)!=len(test_ids) or not np.isin(pred,[-1,1]).all():
        raise ValueError('Prédictions finales invalides.')
    output=result_dir/'submission_nn.csv'
    # Utiliser le helper officiellement fourni avec le projet.
    helpers.create_csv_submission(test_ids,pred,str(output))
    # Vérifier le contenu livré, indépendamment des tableaux ayant servi à écrire.
    loaded=np.loadtxt(output,delimiter=',',skiprows=1,dtype=np.int64)
    if loaded.shape!=(len(test_ids),2) or not np.array_equal(loaded[:,0],test_ids) or not np.array_equal(loaded[:,1],pred):
        raise AssertionError('Le CSV relu ne correspond pas aux IDs/prédictions.')
    np.save(result_dir/'test_scores.npy',scores)
    write_json(result_dir/'submission_info.json',dict(rows=len(pred),positives=int(np.sum(pred==1)),
                threshold=selection['threshold'],sha256=file_hash(output),submitted=False))
    print(f'CSV prêt : {output}',flush=True)
    return output



def run_single_baseline(project_dir,cache_dir,result_dir,baseline_config=None):
    """Un seul MLP : split -> seuil/early stopping -> test -> CSV.

    baseline_config peut remplacer hidden_size, learning_rate, l2,
    positive_weight, batch_size, epochs, seed, clip et dtype.
    """
    project_dir=Path(project_dir).resolve()
    result_dir=Path(result_dir).resolve()
    result_dir.mkdir(parents=True,exist_ok=True)
    if (result_dir/'selection.json').exists():
        raise FileExistsError('Une expérience existe déjà : choisir un nouveau dossier de résultats.')
    shared=read_shared(project_dir)
    helpers=read_project_module(project_dir,'helpers.py','brfss_nn_helpers')
    x,y,ids,names=load_training_data(project_dir,cache_dir)
    folds=stratified_k_fold(y,seed=SEED)
    development,holdout=folds[0]
    local_train,local_tune=stratified_train_val_split(y[development],val_ratio=.2,seed=SEED+1)
    train,tuning=development[local_train],development[local_tune]
    splits=dict(train=train,tuning=tuning,development=development,holdout=holdout)
    np.savez_compressed(result_dir/'experiment_splits.npz',**splits)
    # Baseline simple : un seul réseau, hyperparamètres fixés, aucun ensemble.
    config=dict(name='mlp32_baseline',hidden_size=32,learning_rate=.001,l2=.001,
                positive_weight=1.,batch_size=512,epochs=30,seed=SEED,clip=10.,dtype='float32')
    if baseline_config is not None:
        allowed={'hidden_size','learning_rate','l2','positive_weight','batch_size',
                 'epochs','seed','clip','dtype'}
        unknown=set(baseline_config)-allowed
        if unknown:
            raise ValueError(f'Paramètres baseline inconnus : {sorted(unknown)}')
        config.update(baseline_config)
    config['name']=f"mlp{config['hidden_size']}_baseline"
    a,b=prepare(x[train],x[tuning],names,shared)
    model=BinaryMLP(**{k:v for k,v in config.items() if k!='name'})
    def monitor(candidate):
        score=candidate.predict_scores(b)
        threshold,f1=best_f1_threshold(y[tuning],score)
        metrics=binary_metrics(y[tuning],(score>=threshold).astype(int))
        return dict(selection_score=f1,threshold=threshold,accuracy=metrics['accuracy'])
    print(f"Baseline unique : MLP {config['hidden_size']} neurones ; train={len(train)}, réglage={len(tuning)}, holdout={len(holdout)}",flush=True)
    model.fit(a,y[train],monitor=monitor,patience=5,verbose=True)
    scores=model.predict_scores(b)
    threshold,tuning_f1=best_f1_threshold(y[tuning],scores)
    config['epochs']=model.best_epoch_
    config['name']='mlp32_baseline'
    row=dict(name=config['name'],config=config,best_epoch=model.best_epoch_,
             threshold=threshold,tuning=binary_metrics(y[tuning],(scores>=threshold).astype(int)),
             tuning_at_05=binary_metrics(y[tuning],(scores>=.5).astype(int)),checkpoint='mlp32_baseline.npz')
    model.save(result_dir/row['checkpoint'])
    write_json(result_dir/'mlp32_baseline_history.json',model.history_)
    write_json(result_dir/'tuning_trials.json',[row])
    selection=dict(kind='single_neural_network_baseline',
        members=[dict(name=config['name'],config={k:v for k,v in config.items() if k!='name'})],
        threshold=threshold,tuning_metrics=row['tuning'],
        preprocess_options=PREPROCESS_OPTIONS,
        selection_rule='one fixed architecture (32 hidden units); early stopping and threshold use tuning data only',
        outer_seed=SEED,inner_seed=SEED+1,
        sample_counts=dict(train=len(train),tuning=len(tuning),holdout=len(holdout)))
    manifest=dict(project_dir=str(project_dir),numpy_version=np.__version__,
        shared_hashes={name:file_hash(project_dir/name) for name in ('preprocessing.py','validation.py','helpers.py')},
        source_hashes={p:file_hash(Path(__file__).parent/p) for p in ('nn_model.py','run_nn.py')},
        rows=len(y),positive_count=int(y.sum()),feature_names=names,
        train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        data_signatures={p:dict(size=(project_dir/'data'/'dataset'/p).stat().st_size,
                                mtime_ns=(project_dir/'data'/'dataset'/p).stat().st_mtime_ns)
                         for p in ('x_train.csv','y_train.csv','x_test.csv')})
    write_json(result_dir/'manifest.json',manifest)
    write_json(result_dir/'selection.json',selection)
    print(f"Seuil figé={threshold:.6f}; F1 réglage={tuning_f1:.4f}; époque={model.best_epoch_}",flush=True)
    evaluate_frozen(x,y,names,shared,result_dir,selection)
    generate_submission(project_dir,cache_dir,result_dir,x,y,ids,names,shared,helpers,selection)
    return selection


def run_experiment(project_dir,cache_dir,result_dir,stage='all'):
    project_dir=Path(project_dir).resolve()
    result_dir=Path(result_dir).resolve()
    result_dir.mkdir(parents=True,exist_ok=True)
    shared=read_shared(project_dir)
    helpers=read_project_module(project_dir,'helpers.py','brfss_nn_helpers')
    x,y,ids,names=load_training_data(project_dir,cache_dir)
    hashes={name:file_hash(project_dir/name) for name in ('preprocessing.py','validation.py','helpers.py')}
    if stage=='baseline':
        return run_single_baseline(project_dir,cache_dir,result_dir)
    if stage in ('all','tune'):
        if (result_dir/'selection.json').exists():
            raise FileExistsError('Une sélection existe déjà : choisir un nouveau dossier pour une nouvelle expérience.')
        manifest=dict(project_dir=str(project_dir),numpy_version=np.__version__,shared_hashes=hashes,
                      source_hashes={p:file_hash(Path(__file__).parent/p) for p in ('nn_model.py','run_nn.py')},
                      rows=len(y),positive_count=int(y.sum()),feature_names=names,
                      train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                      data_signatures={p:dict(size=(project_dir/'data'/'dataset'/p).stat().st_size,
                                             mtime_ns=(project_dir/'data'/'dataset'/p).stat().st_mtime_ns)
                                       for p in ('x_train.csv','y_train.csv','x_test.csv')})
        write_json(result_dir/'manifest.json',manifest)
        selection=tune_models(x,y,names,shared,result_dir)
    else:
        manifest=json.loads((result_dir/'manifest.json').read_text())
        if manifest['shared_hashes']!=hashes:
            raise ValueError('Les modules partagés ont changé depuis le réglage.')
        if manifest['train_ids_sha256']!=hashlib.sha256(ids.tobytes()).hexdigest():
            raise ValueError('L’ordre des observations a changé depuis le réglage.')
        current_signatures={p:dict(size=(project_dir/'data'/'dataset'/p).stat().st_size,
                                   mtime_ns=(project_dir/'data'/'dataset'/p).stat().st_mtime_ns)
                            for p in ('x_train.csv','y_train.csv','x_test.csv')}
        if current_signatures!=manifest['data_signatures']:
            raise ValueError('Les fichiers de données ont changé depuis le réglage.')
        current_sources={p:file_hash(Path(__file__).parent/p) for p in ('nn_model.py','run_nn.py')}
        if manifest['source_hashes']!=current_sources:
            raise ValueError('Le code du réseau/runner a changé depuis le réglage.')
        selection=json.loads((result_dir/'selection.json').read_text())
    if stage in ('all','evaluate'):
        evaluate_frozen(x,y,names,shared,result_dir,selection)
    if stage in ('all','final'):
        generate_submission(project_dir,cache_dir,result_dir,x,y,ids,names,shared,helpers,selection)
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=Path('.'))
    parser.add_argument('--cache-dir',type=Path,default=Path('work/nn_cache'))
    parser.add_argument('--result-dir',type=Path,default=Path('nn_results'))
    parser.add_argument('--stage',choices=['baseline','all','tune','evaluate','final'],default='baseline')
    args=parser.parse_args()
    run_experiment(args.project_dir,args.cache_dir,args.result_dir,args.stage)
