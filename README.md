# RegEnsemble
This package is for combining predictions from simple base estimators for improved performance and reduced variation.

Benefits:
- Generalizability: aggregate weak learners for better performance.
- Adaptiveness: explore different proportions of features/samples.
- Robustness: a blend of base linear estimators keep the final model intuitive, yet powerful in reducing dispersion of predictions.

## Important Concepts
- Weighted Average
  - Assign weights to predictions from base-estimators according to out of bag error
  - 𝑤_(𝑏𝑎𝑠𝑒_𝑒𝑠𝑡𝑖𝑚𝑎𝑡𝑜𝑟)=𝑒^(− 𝑒𝑟𝑟𝑜𝑟)
*error: normalized out of sample RMSE*

- Weighted Sampling
  - Assign weights to the subset of feature selection
  - Coefficients as sampling weights

- Model Selection
  - Select base estimators with error lower than existing estimators in the initial pool

## Methods
- `fit(X,y)`: Fit ensemble model.
- `predict(X)`: Predict using the ensemble model.
- `score(X,y)`: Return the coefficient of determination of the prediction.
- `feature_importances_([,reset])`: Retrieve coefficient-based feature importance. 

## Parameters
- `estimator` : regression object or list of objects, default = LinearRegression()
    - If `estimator` is a regression object, it is taken as the base estimator and will be replicated `n_estimators` times to generate the "forest". 
    - If `estimator` is a list of objects, it defines all the "trees" in this "forest". In this case, `n_estimators` is omitted. 
    - Acceptted estimators should have an attribute `coef_` such as: LinearRegression, Ridge, Lasso, etc.

- `n_estimators` : int, default = 1
    Number of trees in the "forest" when `estimator` is a regression object. Omit when `estimator` is a list of objects. 

- `k_top_models` : int, default = -1
    Minimum number of estimators to keep, if -1, use `n_estimaotrs`. 

- `frac_random_samples` : float, default = 1.0
    Fraction of random samples for base estimator. 

- `frac_random_features` : float, default = 1.0
    Fraction of random features for base estimator.

- `random_state` : int, default = 0
    Controls the random resampling of the original dataset (sample wise and feature wise).

## References
De Nard, G., Hediger, S., & Leippold, M. (2020). Subsampled factor models for asset pricing: The rise of Vasa. Journal of Forecasting.

