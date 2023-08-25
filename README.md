# RegEnsemble
This package is for  combining predictions from simple base estimators for improved performance and reduced variation.\
Benefits:
- Generalizability: aggregate weak learners for better performance
- Adaptiveness: explore different proportions of features/samples 
- Robustness: a blend of base linear estimators keep the final model intuitive, yet powerful in reducing dispersion of predictions 

## Important Concepts
### Weighted Average
- Assign weights to predictions from base-estimators according to out of bag error
- 𝑤_(𝑏𝑎𝑠𝑒_𝑒𝑠𝑡𝑖𝑚𝑎𝑡𝑜𝑟)=𝑒^(− 𝑒𝑟𝑟𝑜𝑟)
*error: normalized out of sample RMSE*
### Weighted Sampling
- Assign weights to the subset of feature selection
- Coefficients as sampling weights
### Model Selection
- Select base estimators with error lower than existing estimators in the initial pool

## Methods
- `fit(X,y)`: Build a bunch of regressors from the training set (X, y).
- `predict(X)`: Predict regression target for X.
- `evaluate()`: Evaluate the model by calculating out-of-bag error.
- `feature_importances_()`: The coefficient-based feature importance. 
- `set_params()`
## Parameters
`model`: string \
Base estimator *{’LinearRegression’, ‘Lasso’, ’ElasticNet’, ’Ridge’, ’SVR’,}*. The default is `ElasticNet`. \
`n_estimators` : integer \   
Number of estimators. The default is 20. \
`patience`: integer \
Minimum number of estimators to keep at the beginning of the training, if `None`, use `n_estimaotrs`. The default is `None`. \ 
`frac_random_sample` : float \
Fraction of data for each base model. The default is 0.75. \
`frac_random_feature` : float \
Fraction of features for each base model, if `None`, randomly select sqrt of the features. The default is `None`. \
`random_state`: integer \
Controls the random resampling of the original dataset (sample wise and feature wise), if `None`, 0. The default is None. \
`kwargs`: additional hyperparameters
