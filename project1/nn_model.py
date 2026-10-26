"""Petit réseau binaire NumPy : ReLU, BCE pondérée, L2 et Adam.

Une seule couche cachée ; hidden_size=0 donne une baseline logistique.
Aucune bibliothèque ML. Toutes les graines sont locales au modèle.
Les gradients sont ceux de mean(weight * BCE) + l2/2 * sum(weights**2).
Les biais ne sont pas régularisés. Les poids positifs modifient la calibration :
la sortie sigmoid est un score à seuiller, pas une probabilité médicale validée.
"""
import json
from pathlib import Path
import numpy as np


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def _labels(y, n):
    y = np.asarray(y)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all():
        raise ValueError('y doit être un vecteur 0/1 aligné sur x.')
    return y


class BinaryMLP:
    """Une couche cachée ReLU avec sorties vectorielles (n,).

    clip=10 borne les entrées déjà standardisées, sans statistique apprise
    supplémentaire. Adam reçoit des mini-batches mélangés à chaque époque.
    fit() réinitialise toujours les poids : aucun état entre deux plis.
    """
    def __init__(self, hidden_size=32, learning_rate=1e-3, l2=1e-4,
                 positive_weight=1.0, batch_size=512, epochs=25, seed=0,
                 clip=10.0, dtype='float32'):
        for name, value, lower in [('hidden_size', hidden_size, 0),
                                   ('batch_size', batch_size, 1), ('epochs', epochs, 1),
                                   ('seed', seed, 0)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f'{name} doit être un entier >= {lower}.')
        for name, value in [('learning_rate', learning_rate), ('positive_weight', positive_weight), ('clip', clip)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} doit être strictement positif et fini.')
        if not np.isfinite(l2) or l2 < 0 or dtype not in ('float32', 'float64'):
            raise ValueError('l2 doit être >=0 et dtype float32 ou float64.')
        self.config = dict(hidden_size=int(hidden_size), learning_rate=float(learning_rate),
            l2=float(l2), positive_weight=float(positive_weight), batch_size=int(batch_size),
            epochs=int(epochs), seed=int(seed), clip=float(clip), dtype=dtype)
        self.params = None

    def _x(self, x):
        a = np.asarray(x, dtype=self.config['dtype'])
        if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
            raise ValueError('Le réseau attend une matrice finie, imputée et standardisée.')
        if self.params is not None and a.shape[1] != self.n_features_:
            raise ValueError('Nombre de features incompatible avec le réseau.')
        return np.clip(a, -self.config['clip'], self.config['clip'])

    def _initialize(self, n_features, prevalence):
        rng = np.random.default_rng(self.config['seed'])
        h = self.config['hidden_size']
        dtype = self.config['dtype']
        # Initialiser le biais près de l'optimum d'une constante pondérée.
        p = np.clip(prevalence, 1e-6, 1-1e-6)
        bias = np.log(p*self.config['positive_weight']/(1-p))
        if h:
            self.params = dict(
                w1=(rng.normal(size=(n_features,h))*np.sqrt(2/n_features)).astype(dtype),
                b1=np.zeros(h,dtype=dtype),
                w2=(rng.normal(size=h)*0.1/np.sqrt(h)).astype(dtype),
                b2=np.array([bias],dtype=dtype))
        else:
            self.params = dict(w2=np.zeros(n_features,dtype=dtype), b2=np.array([bias],dtype=dtype))
        self.n_features_ = n_features
        return rng

    def _forward(self, x):
        p = self.params
        if self.config['hidden_size']:
            hidden = np.maximum(x @ p['w1'] + p['b1'], 0)
        else:
            hidden = x
        return hidden @ p['w2'] + p['b2'][0], hidden

    def _loss_grad(self, x, y):
        """x est déjà préparé ; utilisé aussi par les tests de gradient numérique."""
        z, hidden = self._forward(x)
        weight = np.where(y == 1, self.config['positive_weight'], 1.0).astype(x.dtype)
        loss = np.mean(weight * ((1-y)*np.logaddexp(0., z) + y*np.logaddexp(0., -z)))
        delta = (sigmoid(z)-y)*weight/len(y)
        l2 = self.config['l2']
        gradients = dict(w2=hidden.T @ delta + l2*self.params['w2'],
                         b2=np.array([delta.sum()],dtype=x.dtype))
        if self.config['hidden_size']:
            dh = delta[:,None]*self.params['w2'][None,:]
            dh *= hidden > 0
            gradients['w1'] = x.T @ dh + l2*self.params['w1']
            gradients['b1'] = dh.sum(axis=0)
        penalty = sum(float(np.sum(v*v)) for k,v in self.params.items() if k.startswith('w'))
        return float(loss + 0.5*l2*penalty), gradients

    def predict_scores(self, x, batch_size=8192):
        if self.params is None:
            raise ValueError('Entraîner ou charger le réseau avant predict_scores.')
        if not isinstance(batch_size, (int,np.integer)) or batch_size < 1:
            raise ValueError('batch_size doit être un entier positif.')
        a = self._x(x)
        out = np.empty(len(a), dtype=float)
        for start in range(0,len(a),batch_size):
            z,_ = self._forward(a[start:start+batch_size])
            out[start:start+batch_size] = sigmoid(z)
        if not np.isfinite(out).all():
            raise FloatingPointError('Scores non finis : réduire learning_rate ou inspecter les données.')
        return out

    def fit(self, x, y, *, monitor=None, patience=5, min_delta=1e-4, verbose=False):
        """Entraîner ; monitor(model) -> dict contenant 'selection_score'.

        Le monitor doit utiliser uniquement votre jeu de réglage. Le checkpoint
        choisi maximise selection_score ; patience n'agit que si monitor existe.
        Sans monitor, toutes les époques demandées sont effectuées.
        L'historique et le meilleur nombre d'époques sont conservés.
        """
        if not isinstance(patience,(int,np.integer)) or patience < 1 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError('patience >=1 et min_delta >=0 requis.')
        # Un nouvel appel fit est un entraînement neuf, même si la dimension change.
        self.params = None
        x = self._x(x)
        y = _labels(y,len(x)).astype(x.dtype)
        if len(np.unique(y)) != 2:
            raise ValueError('Les deux classes doivent être présentes dans le train.')
        rng = self._initialize(x.shape[1],float(y.mean()))
        first = {k:np.zeros_like(v) for k,v in self.params.items()}
        second = {k:np.zeros_like(v) for k,v in self.params.items()}
        step, wait, best_score, best = 0, 0, -np.inf, None
        significant_best = -np.inf
        self.history_ = []
        self.best_epoch_ = self.config['epochs']
        for epoch in range(1,self.config['epochs']+1):
            permutation = rng.permutation(len(y))
            loss_sum = 0.0
            for start in range(0,len(y),self.config['batch_size']):
                idx = permutation[start:start+self.config['batch_size']]
                loss, gradients = self._loss_grad(x[idx],y[idx])
                if not np.isfinite(loss):
                    raise FloatingPointError('Loss non finie.')
                loss_sum += loss*len(idx)
                step += 1
                for k,grad in gradients.items():
                    if not np.isfinite(grad).all():
                        raise FloatingPointError(f'Gradient non fini : {k}.')
                    first[k] *= .9
                    first[k] += .1*grad
                    second[k] *= .999
                    second[k] += .001*grad*grad
                    m = first[k]/(1-.9**step)
                    v = second[k]/(1-.999**step)
                    self.params[k] -= self.config['learning_rate']*m/(np.sqrt(v)+1e-8)
            row = dict(epoch=epoch, training_batch_loss=loss_sum/len(y))
            if monitor is not None:
                metrics = monitor(self)
                score = float(metrics['selection_score'])
                if not np.isfinite(score):
                    raise ValueError('selection_score doit être fini.')
                row.update(metrics)
                if score > best_score:
                    best_score, self.best_epoch_ = score, epoch
                    best = {k:v.copy() for k,v in self.params.items()}
                if score > significant_best + min_delta:
                    significant_best, wait = score, 0
                else:
                    wait += 1
            self.history_.append(row)
            if verbose:
                suffix = f", tuning_F1={row['selection_score']:.4f}" if monitor is not None else ''
                print(f"epoch {epoch:02d}: batch_loss={row['training_batch_loss']:.5f}{suffix}",flush=True)
            if monitor is not None and wait >= patience:
                break
        if best is not None:
            self.params = best
        return self

    def save(self,path):
        """Poids + config, sans pickle. Le prétraitement partagé reste distinct."""
        if self.params is None:
            raise ValueError('Aucun poids à sauvegarder.')
        metadata=dict(config=self.config,n_features=self.n_features_,best_epoch=self.best_epoch_,history=self.history_)
        with Path(path).open('wb') as f:
            np.savez_compressed(f, metadata=np.array(json.dumps(metadata)), **self.params)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            metadata=json.loads(f['metadata'].item())
            obj=cls(**metadata['config'])
            obj.n_features_=int(metadata['n_features'])
            h=obj.config['hidden_size']
            shapes={'w2':(h or obj.n_features_,),'b2':(1,)}
            if h:
                shapes.update(w1=(obj.n_features_,h),b1=(h,))
            obj.params={k:f[k].copy() for k in shapes}
            if any(obj.params[k].shape != shape or not np.isfinite(obj.params[k]).all() for k,shape in shapes.items()):
                raise ValueError('Checkpoint invalide.')
            obj.best_epoch_=int(metadata['best_epoch'])
            obj.history_=metadata['history']
            return obj
