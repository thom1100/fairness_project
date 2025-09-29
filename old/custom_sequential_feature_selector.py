import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from mlxtend.feature_selection import SequentialFeatureSelector as MlxtendSFS

class MlxtendSFSWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, k_features="best", forward=True, floating=False,
                 scoring=None, cv=5, n_jobs=1, verbose=0, pre_dispatch="2*n_jobs"):
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        
    def fit(self, X, y=None):
        # On clone l’estimateur pour éviter les effets de bord
        self.estimator_ = clone(self.estimator)
        self.sfs_ = MlxtendSFS(
            self.estimator_,
            k_features=self.k_features,
            forward=self.forward,
            floating=self.floating,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )
        self.sfs_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.sfs_.transform(X)
    
    def get_support(self, indices=False):
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[list(self.sfs_.k_feature_idx_)] = True
        if indices:
            return np.where(mask)[0]
        return mask
    
    @property
    def subsets_(self):
        return self.sfs_.subsets_
    
    @property
    def k_feature_names_(self):
        return self.sfs_.k_feature_names_
