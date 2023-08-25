import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

from joblib import Parallel
from sklearn.utils.fixes import delayed

class RegEnsemble(object):
    '''
    Lasso Ensemble
    '''
    
    def __init__(self, model = 'ElasticNet',n_estimators = 20, patience = None, frac_random_sample = 0.75, frac_random_feature = None,random_state = None,**kwargs):
        '''
        

        Parameters
        ----------
        model: string {'LinearRegression','SVR','Ridge','Lasso','ElasticNet'}
            The default is 'ElasticNet'.
        n_estimators : integer
            Number of estimators. The default is 20.
        patience: integer
            Minimum number of estimators to keep at the beginning of the training, if None, use n_estimaotrs. The default is None. 
        frac_random_sample : float
            Fraction of data for each base model. The default is 0.75.
        frac_random_feature : float
            Fraction of features for each base model, if None, randomly select half of the features. The default is None.
        random_state: integer
            Random seed for the random resampling of the original dataset (sample-wise and feature-wise), if None, 0. The default is None.
        **kwargs : 
            Adjust hyperparameters of the estimator.

        Returns
        -------
        None.

        '''
        # Initialization done here
        self.n_estimators = n_estimators
        self.modelname = model
        if model == 'SVR':
            self.lasso_models = [SVR(kernel='linear',**kwargs) for i in range(n_estimators)]
        elif model == 'Lasso':
            self.lasso_models = [Lasso(**kwargs) for i in range(n_estimators)]
        elif model == 'ElasticNet':
            self.lasso_models = [ElasticNet(**kwargs) for i in range(n_estimators)]
        elif model == 'Ridge':
            self.lasso_models = [Ridge(**kwargs) for i in range(n_estimators)]
        elif model == 'LinearRegression':
            self.lasso_models = [LinearRegression(**kwargs) for i in range(n_estimators)]
        else:
            raise ValueError('Please select base estimators from LinearRegresssion, Lasso, Ridge, ElasticNet, and SVR')
        self.frac_random_sample = frac_random_sample
        self.frac_random_feature = frac_random_feature
        self.patience = n_estimators if patience is None else patience
        self.random_state = 0 if random_state is None else random_state
        self.kwargs = kwargs
        
       
    def fit(self, X, y):
        '''
        

        Parameters
        ----------
        X : data frame or arraly-like of shape (n_samples, n_features), 
        y : ata frame or array-like of shape (n_samples,).

        Returns
        -------
        None.

        '''
        
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).add_prefix('feature_')
        if isinstance(y,pd.Series):
            y = y.to_frame()
        elif not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=['y'])
        
        self.X = X
        self.X_feature_names = X.columns.to_list()
        
        self.n_samples = max(1, int(len(self.X)*self.frac_random_sample))
        if self.frac_random_feature is None:
            self.n_features = max(1,int(len(self.X_feature_names)**0.5))
        else:
            self.n_features = max(1,int(len(self.X_feature_names)*self.frac_random_feature))
        self.y = y
        self.y_feature_names = y.columns[0]
        mwt_all=[]
        # Initializing the sample datasets for each estimator
        self._random_dataset = self.__random_sample__()
        
        for i, mdl in enumerate(self.lasso_models):
            model_data = self._random_dataset[f'sample-{i}']
            mdl.fit(self.X[model_data["selected_features"]].iloc[model_data["selected_samples"]].to_numpy(),       
                    self.y.iloc[model_data["selected_samples"]].to_numpy())
            mwt_all.append(self.__mdlwt__(model_data,mdl))

        mwt_all = mwt_all/np.sum(mwt_all)
        
        for i in range(self.patience,self.n_estimators):
            if mwt_all[i] >= np.min(mwt_all[:self.patience]):
                mwt_all[i] = np.inf 
        
        self.mwt_all = mwt_all.copy()
        self.mwt = [np.exp(-x) for x in mwt_all]
        
        
    def refine(self,dev = 0.1):

        '''
        Parameters
        ----------
        dev : float between 0 and 1, how much deviation of error is allowed between the test base estimator and the average level of top estimators (in patience size).
        
        Returns
        -------
        None.

        '''
        sorted_idx = self.mwt_all.argsort()
        err_bs = np.mean(self.mwt_all[sorted_idx][:self.patience])
        
        for i in range(self.n_estimators):
            if (self.mwt_all[i]-err_bs) >= dev*err_bs:
                self.mwt_all[i] = np.inf
        self.mwt = [np.exp(-x) for x in self.mwt_all]
        
        
    def __mdlwt__(self,model_data,mdl):
        
        yhat = mdl.predict(self.X[model_data["selected_features"]].iloc[model_data["unselected_samples"]].to_numpy())
        from sklearn.metrics import mean_squared_error
        mwt = mean_squared_error(self.y.iloc[model_data["unselected_samples"]].to_numpy(),yhat, squared=False)
        return mwt
    
    def __ftwt__(self):
        fwt = []
        lasso_models = self.lasso_models.copy()
        all_importances = np.zeros((1,self.X.shape[1]))
        all_importances = pd.DataFrame(all_importances,columns=self.X.columns)
        for i,fmdl in enumerate(lasso_models):
            fmdl.fit(self.X,self.y)
            all_importances+= np.abs(fmdl.coef_)
        all_importances = np.mean(all_importances.to_numpy(), axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def get_cross_validation_scores(self):
        '''
        

        Returns
        -------
        RMSE : float
            Root mean square error
        MAPE : float
            Mean absolute percentage error

        
        OOB: evaluate the model by calculating out-of-bag error.
        Identify the set of estimators that consider the record as an out-of-bag sample.
        Predict using each of the above found estimators.
        Use average of the predictions as the final output  for this record.
        '''
        y_hat = np.empty((len(self.y), len(self.lasso_models)))
        y_hat[:] = np.nan
        for i, mdl in enumerate(self.lasso_models):
            model_data = self._random_dataset[f'sample-{i}']
            X_i = self.X[model_data["selected_features"]].iloc[model_data["unselected_samples"]].to_numpy()
            if self.modelname == 'LinearRegression' or self.modelname == 'Ridge':
                y_hat[model_data["unselected_samples"], i] = mdl.predict(X_i).ravel()
            else:
                y_hat[model_data["unselected_samples"], i] = mdl.predict(X_i)
        
        self.y_hat = np.nanmean(y_hat, axis = 1).reshape(-1,1)
        
        RMSE = np.nanmean((self.y.to_numpy() - self.y_hat)**2)**0.5
        MAPE = np.nanmean(np.abs((self.y.to_numpy()-self.y_hat)/(self.y.to_numpy())))
        return RMSE, MAPE
    
    def predict(self, X):
        '''
        

        Parameters
        ----------
        X : data frame or arraly-like of shape (n_samples, n_features), 
            DESCRIPTION.

        Returns
        -------
        Prediction: data frame or array-like of shape (n_samples,).
           

        '''
        # prediction of new record/records
        if isinstance(X,pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).add_prefix('feature_')
        #print('X',X)
        y_pred = np.zeros((X.shape[0], len(self.lasso_models)))
        for i,mdl in enumerate(self.lasso_models):
            model_data = self._random_dataset[f'sample-{i}']
            X_i = X[model_data["selected_features"]].to_numpy()
            y_pred[:,i] = mdl.predict(X_i).ravel()
        return np.average(y_pred,weights = self.mwt,axis = -1)#y_pred.mean(axis = -1)#.reshape(1,-1)
    
    def get_params(self, deep=True):
        return {"model": self.modelname,
            "n_estimators": self.n_estimators, "patience": self.patience,
                "frac_random_sample": self.frac_random_sample,"frac_random_feature": self.frac_random_feature,
                "random_state": self.random_state,}
                #"**kwargs": self.kwargs}#,"imp":self.imp}
    

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def __random_sample__(self):
        #Create a sample dataset by random sampling from the original dataset X and y.
        random_sample = {}
        allindex = [x for x in range(len(self.X))]
        
        prob = self.__ftwt__()
        
        
        
        for i in range(self.n_estimators):
            random_sample[f'sample-{i}'] = {}
            # random select fraction of data sample
            # random select fraction of features
            np.random.seed(10*i+self.random_state)
            
            random_sample[f'sample-{i}']["selected_features"] = np.random.choice(self.X_feature_names,
                                                                                 self.n_features, 
                                                                                 replace=False,p=prob).tolist()#
            random_sample[f'sample-{i}']["selected_samples"] = np.sort(np.random.choice([x for x in range(len(self.X))], 
                                                                               self.n_samples,
                                                                               replace=False)).tolist()
            random_sample[f'sample-{i}']["unselected_samples"] = list(set(allindex) - set(random_sample[f'sample-{i}']["selected_samples"]))
            
        return random_sample
    
    def feature_importances_(self):
        """
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        
        all_importances = np.zeros((1,self.X.shape[1]))
        all_importances = pd.DataFrame(all_importances,columns=self.X.columns)
        for i, mdl in enumerate(self.lasso_models):
            model_data = self._random_dataset[f'sample-{i}']
            all_importances[model_data["selected_features"]]+= np.abs(mdl.coef_)
        
        
        all_importances = np.mean(all_importances.to_numpy(), axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)