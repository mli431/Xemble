import numpy as np
import pandas as pd
import warnings
from numbers import Integral
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.base import (
    clone,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.utils.validation import check_is_fitted, DataConversionWarning
#from sklearn.execpetions import DataConversionWarning

warnings.filterwarnings('ignore', category=DataConversionWarning)

class EnsembleRegressor(MultiOutputMixin, RegressorMixin):
    """
    This package is for combining predictions from simple base estimators for improved performance and reduced variation.

    Parameters
    ----------
    estimator : regression object or list of objects, default = LinearRegression()
        If `estimator` is a regression object, it is taken as the base estimator and will be 
        replicated `n_estimators` times to generate the "forest". 
        If `estimator` is a list of objects, it defines all the "trees" in this "forest". In 
        this case, `n_estimators` is omitted. 
        Accepted estimators should possess a `coef_` attribute, such as: 
            LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge, TweedieRegressor, ElasticNet
    
    n_estimators : int, default = 1
        Number of "trees" in the "forest" when `estimator` is a regression object. Omitted when 
        `estimator` is a list of objects. 
    
    k_top_models : int, default = -1
        Minimum number of estimators to keep, if -1, use `n_estimaotrs`. 

    frac_random_samples : float, default = 1.0
        Fraction of random samples for each base estimator. 
    
    frac_random_features : float, default = 1.0
        Fraction of random features for each base estimator.
    
    random_state : int, default = 0
        Controls the randomness of resampling of the original dataset (sample-wise and feature-wise).
    
    
    Attributes
    ----------
    estimators_ : list
        The list of base estimators.
    
    oob_rmse_ : list
        Root-mean-squared-error of the base estimators for out-of-bag samples.
    
    model_weights_ : list
        Weights of each base estimator after fitting. 
    
    n_fit_samples_ : int
        Number of samples used to fit each estimator.
    
    n_fit_features_ : int
        Number of features used to fit each estimator.
    
    random_samples_ : list
        The list of random sample index used in each estimator in `estimators_` in shape (n_fit_samples_, ).
    
    oob_samples_ : list
        The list of out-of-bag sample index for each estimator in `estimators_` that can be used to 
        evaluate the out-of-bag performance.
    
    random_features_ : list
        The list of random features used in each estimator in `estimators_` in shape (n_fit_features_, ).
    
    
    Notes
    -----
    From the implementation point of view, this is a modified version of random-forest using linear models 
    as base estimators (trees). 

    Examples
    --------
    >>> From RegEnsemble import EnsembleRegressor
    >>> From sklearn.linear_model import LinearRegression
    >>> reg = EnsembleRegressor(estimator = LinearRegression(),
    >>>                         n_estimators = 100,
    >>>                         k_top_models = 50,
    >>>                         frac_random_samples = 0.8, 
    >>>                         frac_random_features = 0.8).fit(X, y).enhance(0.5)
    >>> reg.score(X, y)
    
    Reference
    ---------
    De Nard, G., Hediger, S., & Leippold, M. (2020). Subsampled factor models for asset pricing: The rise of Vasa. Journal of Forecasting.

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
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.k_top_models = k_top_models
        self.frac_random_samples  = frac_random_samples
        self.frac_random_features = frac_random_features
        self.random_state = random_state
        
        
    def fit(self, X, y):
        """
        Fit EnsembleRegressor model.

        Parameters
        ----------
        X : array-like matrix of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        # formatting estimators
        if isinstance(self.estimator, list) and all([isinstance(mdl, object) for mdl in self.estimator]):
            estimators = self.estimator.copy()
        elif isinstance(self.estimator, object) and isinstance(self.n_estimators, int) and self.n_estimators > 0:
            estimators = [clone(self.estimator) for _ in range(self.n_estimators)]
        else:
            raise ValueError('estimators should be an object or list, n_estimators should be integral.' + 
                             f'Input estimator is {type(self.estimator)} and n_estimators is {type(self.n_estimators)}')
        np.random.seed(self.random_state)
        self.estimators_ = np.random.permutation(estimators)
        self.n_estimators_ = len(self.estimators_)

        # cast the input into pd.DataFrame for X, pd.Series for y
        self.X = X
        self.y = y
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

        # fitting number of samples and features
        self.n_samples, self.n_features = self.X_.shape
        
        self.feature_names_ = self.X_.columns.to_list()
        
        self.n_fit_samples_ = max(int(self.n_samples*self.frac_random_samples), 1)
        self.n_fit_features_ = max(int(self.n_features * self.frac_random_features), 1)

        # random features and samples
        self.random_features_, self.random_samples_, self.oob_samples_ = self._random_choice()

        # fit each estimator, calculate the out-of-bag RMSE
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
        
        # select k top models by assigning model_weights_
        if self.k_top_models <= 0:
            self.k_top_models = self.n_estimators_

        sum_ = np.sum(self.oob_rmse_)
        min_ = -1.0/min(self.oob_rmse_[:self.k_top_models])
        self.model_weights_ = [np.exp((sum_-x)/sum_) if (i <= self.k_top_models)or(x < min_) 
                               else 0.0 for i, x in enumerate(self.oob_rmse_)]
        return self

    def enhance(self, quantile):
        """ 
        Enhance the model by updating the model_weights_ and keep those models with largest 
        weights (smallest oob-error), i.e., model weights smaller than threshold will be set 
        to 0. 
        
        Parameters
        ----------
        quantile : float between 0.0 to 1.0.
            The quantile value to define the threshold for truncation. Larger value results 
            in a more aggressive enhancement, but may cause over-fitting.
        
        Return
        ------
        self : object
            Enhanced Estimator. 
        """

        weights = [w for w in self.model_weights_ if w > 0]
        threshold = min(max(weights), 
                        np.quantile(weights, quantile)) # keep at least one estimators
        self.model_weights_ = [0 if x < threshold else x for x in self.model_weights_]
        return self
    
    def predict(self, X):
        """
        Predict using the ensemble model.

        Parameters
        ----------
        X : array-like matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Returns predicted values.
        """
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
        
    def _feature_importance(self, reset = False):
        """
            Retrieve the feature importance for the whole ensemble model in an order of `feature_names_`. 

            Parameters
            ----------            
            reset : bool, default = False
                Use True in the `fit` process. 
            
            Returns
            -------
            fi_avg_scale : ndarray of shape (n_fit_features_, )
        """
        
        fi_ = {feature: np.zeros(self.n_estimators_) for feature in self.feature_names_}
        if reset:
            features_in = [self.feature_names_] * self.n_estimators_
            weights = np.ones(self.n_estimators)
        else:
            features_in = self.random_features_
            weights = self.model_weights_
        
        for i, (estimator, feature_in) in enumerate(zip(self.estimators_, features_in)):
            coef = self._get_estimator_coefficients(estimator, feature_in, reset = reset)
            for j, feature in enumerate(feature_in):
                fi_[feature][i] = coef[j]
        
        fi_avg = np.array([np.average(x, weights = weights) for x in fi_.values()])
        fi_avg_scale = fi_avg/fi_avg.sum()
        return fi_avg_scale
    

    def _get_estimator_coefficients(self, estimator, feature_in, reset = False):
        """
            Helper function to retrieve absolute coefficients of an estimator corresponding to `feature_in`.

            Parameter
            ---------
            estimator : object
                Regression object has an attribute `coef_`.
            
            feature_in : list
                Features to be considered in the estimation.
            
            reset : reset : bool, default = False
                Use True in the `fit` process. 
            Return
            ------
            coef : list
                List of regression coefficients for each features in feature_in.
        """
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
    
    
    def _random_choice(self):
        """
            Helper function to do random selection of features and samples for each estimators.
        """
        ibf, inb, oob = [], [], []
        feature_weights = self._feature_importance(reset = True)
        for i in range(self.n_estimators_):
            np.random.seed(self.random_state + i + 1)
            ibf.append(np.random.choice(self.feature_names_, self.n_fit_features_, replace = False, p = feature_weights).tolist())
            inb.append(np.random.choice(self.X_.index, self.n_fit_samples_, replace = False).tolist())
            oob.append([x for x in self.X_.index if x not in inb[-1]])
        return ibf, inb, oob