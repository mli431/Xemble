import numpy as np
import pandas as pd
from numbers import Integral
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.base import (
    clone,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.utils.validation import check_is_fitted

class EnsembleRegressor(MultiOutputMixin, RegressorMixin):
    """
    Ensemble linear regression.

    EnsembleRegressor fits a set of linear regression models w
    
    """
    _parameter_constraints: dict = {
        "estimator": ["object", "list"],
        "n_estimators": [Integral],
        "k_top_models": [Integral],
        "frac_random_samples": [float],
        "frac_random_features": [float],
        "random_state": [Integral]
    }
    def __init__(self, 
                 estimator = LinearRegression(),
                 n_estimators = 1,
                 k_top_models = -1, 
                 frac_random_samples = 1.0, 
                 frac_random_features = 1.0,
                 random_state = 0,
                ):
        """
        Sample Code:

from sklearn.linear_model import Ridge, Lasso, LinearRegression, LassoLars, BayesianRidge, TweedieRegressor, ElasticNet
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
X, y = make_regression(n_samples=2500, n_features=200, n_informative=15, n_targets=1, noise=1, random_state=42)
n_estimators = 100
np.random.seed(0)
pool = [Ridge(), Lasso(), LinearRegression(), ElasticNet()]
estimators = []
for _ in range(n_estimators):
    estimators.append(clone(np.random.choice(pool, 1)[0]))
mdl = EnsembleRegressor(estimator = estimators,
                        n_estimators = n_estimators,
                        k_top_models = 100,
                        frac_random_samples = 0.8,
                        frac_random_features = 0.6,
                        random_state = 42)
mdl.fit(X, y)
yh = mdl.predict(X)
rmse = mean_squared_error(y, yh, squared = False)
plt.scatter(y, yh, marker = '.', label = str(round(rmse,1)) + "("+str(sum(np.array(mdl.model_weights_) > 0)) + ")")
for i in range(5):
    mdl.enhance(0.4)
    yh = mdl.predict(X)
    rmse = mean_squared_error(y, yh, squared = False)
    plt.scatter(y, yh, marker = '.', label = str(round(rmse,1)) + "("+str(sum(np.array(mdl.model_weights_) > 0)) + ")")
plt.legend()
plt.show()
        
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.k_top_models = k_top_models
        self.frac_random_samples  = frac_random_samples
        self.frac_random_features = frac_random_features
        self.random_state = random_state
        
    def fit(self, X, y):
        self.X = X
        self.y = y

        if isinstance(self.estimator, list) and all([isinstance(mdl, object) for mdl in self.estimator]):
            estimators = self.estimator.copy()
        elif isinstance(self.estimator, object) and isinstance(self.n_estimators, int) and self.n_estimators > 0:
            estimators = [clone(self.estimator) for _ in range(self.n_estimators)]
        else:
            raise ValueError('estimators should be an object or list, n_estimators should be integral.' + 
                             f'Input estimator is {type(self.estimator)} and n_estimators is {type(self.n_estimators)}')
        np.random.seed(self.random_state)
        self.estimators_ = np.random.permutation(estimators)
        self.n_estimators_ = len(self.estimators_);
        
        self.model_weights_ = np.ones(self.n_estimators_)
        
        if not hasattr(self.X, 'loc'):
            self.X_ = pd.DataFrame(data = self.X).add_prefix('feature_')
        elif not isinstance(self, pd.DataFrame):
            self.X_ = self.X.copy().to_frame()
        else:
            self.X_ = self.X.copy()

        if not hasattr(self.y, 'loc'):
            self.y_ = pd.DataFrame(data = self.y, columns = ['y'])
        elif not isinstance(self, pd.DataFrame):
            self.y_ = self.y.copy().squeeze()
        else:
            self.y_ = self.y.copy()

        self.n_samples, self.n_features = self.X_.shape
        
        self.feature_names_ = self.X_.columns.to_list()
        
        self.n_samples_ = max(int(self.n_samples*self.frac_random_samples), 1)
        self.n_features_ = max(int(self.n_features * self.frac_random_features), 1)

        self.random_features_, self.random_samples_, self.oob_samples_ = self._random_choice()

        # fit 
        self.fitted_estimators_, self.oob_rmse_ = [], []
        for i in range(self.n_estimators_):
            mdl = clone(self.estimators_[i])
            
            X_inb = self.X_.loc[self.random_samples_[i], self.random_features_[i]]
            y_inb = self.y_.iloc[self.random_samples_[i]]
            
            X_oob = self.X_.loc[self.oob_samples_[i], self.random_features_[i]]
            y_oob = self.y_.iloc[self.oob_samples_[i]]
            
            mdl.fit(X_inb, y_inb)
            self.fitted_estimators_.append(mdl)
            
            yh_oob = mdl.predict(X_oob).reshape(-1,1)
            
            self.oob_rmse_.append(mean_squared_error(y_oob, yh_oob, squared=False))
        
        # n best models
        if self.k_top_models <= 0:
            self.k_top_models = self.n_estimators_

        sum_ = np.sum(self.oob_rmse_)
        min_ = -1.0/min(self.oob_rmse_[:self.k_top_models])
        self.model_weights_ = [np.exp((sum_-x)/sum_) if (i <= self.k_top_models)or(x < min_) 
                               else 0.0 for i, x in enumerate(self.oob_rmse_)]
        return self

    def enhance(self, quantile):
        """
        estimators with highest quantile weights, weights <= quantile will be set to 0
        """
        weights = [w for w in self.model_weights_ if w > 0]
        threshold = min(max(weights), 
                        np.quantile(weights, quantile)) # keep at least one estimators
        self.model_weights_ = [0 if x <= threshold else x for x in self.model_weights_]
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        if not hasattr(X, 'loc'):
            X = pd.DataFrame(data = X.copy()).add_prefix('feature_')
        elif not isinstance(X, pd.DataFrame):
            X = X.copy().to_frame()

        Y_pred = np.asarray(
            [mdl.predict(X[self.random_features_[i]]).ravel() 
             for i, mdl in enumerate(self.fitted_estimators_)]
        ).T
        
        y_pred = np.average(Y_pred, 
                            weights = np.tile(self.model_weights_, 
                                              (X.shape[0], 1)), 
                            axis = 1)
        return y_pred
        
        
    def _get_estimator_coefficients(self, estimator, feature_in, reset):
        estimator = clone(estimator)
        if reset:
            estimator.fit(self.X_[feature_in], self.y_)
        
        if hasattr(estimator, 'coef_'):
            if len(estimator.coef_.shape) == 2:
                coef = abs(estimator.coef_[0])
            else:
                coef = abs(estimator.coef_)
            if len(coef) == len(feature_in):
                return coef
            else:
                raise ValueError(f'length of {estimator.__class__.__name__} coefficients is {len(coef)}, needs {len(feature_in)}')
        else:
            raise NotImplementedError(f'estimator {estimator.__class__.__name__} does not have attribute "coef_" to extract importance')
    
    def _feature_importance(self, estimators, features_in, weights, reset):
        fi_ = {feature: np.zeros(self.n_estimators_) for feature in self.feature_names_}
        for i, (estimator, feature_in) in enumerate(zip(estimators, features_in)):
            coef = self._get_estimator_coefficients(estimator, feature_in, reset)
            for j, feature in enumerate(feature_in):
                fi_[feature][i] = coef[j]
        fi_avg = np.array([np.average(x, weights = weights) for x in fi_.values()])
        return fi_avg/fi_avg.sum()
    
    
    def _random_choice(self):
        ibf, inb, oob = [], [], []
        feature_weights = self._feature_importance(self.estimators_, 
                                                   [self.feature_names_] * self.n_estimators_, 
                                                   np.ones(self.n_estimators_), True)
        for i in range(self.n_estimators_):
            np.random.seed(self.random_state + i + 1)
            ibf.append(np.random.choice(self.feature_names_, self.n_features_, replace = False).tolist())
            inb.append(np.random.choice(self.X_.index, self.n_samples_, replace = False).tolist())
            oob.append([x for x in self.X_.index if x not in inb[-1]])
        return ibf, inb, oob