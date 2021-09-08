import pandas as pd 
import numpy as np 
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import shap
import xgboost as xgb

# Setup
area = 'CE'
data_version = '2021-09-08'
target = 'f_rocof'

# Define folders
data_folder = './data/{}/version-{}/'.format(area,data_version)
res_folder = './results/model_fit/{}/version-{}/target_{}/'.format(area,data_version, target)
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
    
# Load target data
y_train = pd.read_hdf(data_folder+'y_train.h5').loc[:, target]
y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:, target]
if os.path.exists(res_folder+'y_pred_test.h5'):
    y_pred_test = pd.read_hdf(res_folder+'y_pred_test.h5') 
else:
    y_pred_test = pd.read_hdf(data_folder+'y_pred_test.h5') #contains only time index

X_train = pd.read_hdf(data_folder+'X_train.h5')
X_test = pd.read_hdf(data_folder+'X_test.h5')
X = pd.read_hdf(data_folder+'X.h5')


# Daily profile and mean predictor
daily_profile = y_train.groupby(X_train.index.time).mean()
y_pred_test['daily_profile'] = [daily_profile[time] for time in y_test.index.time]
y_pred_test['mean_predictor'] = y_train.mean()


# Gradient boosting Regressor CV hyperparameter optimization

X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train,
                                                                            test_size=0.2)

params_grid = {
    'max_depth':[3,5,7,9,11],
    'learning_rate':[0.01,0.05,0.1, 0.2],
    'subsample': [1,0.7,0.4,0.1] ,
    'reg_lambda':[ 0.1, 1, 10, 50],
    'min_child_weight':[1,5,10,25,50]
}


fit_params = {
    'eval_set':[(X_train_train, y_train_train),(X_train_val, y_train_val)],
    'early_stopping_rounds':20, 
    'verbose':0
}

grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000,
                                            verbosity=0,n_jobs=1),
                           params_grid, verbose=1, n_jobs=7, cv=5)

grid_search.fit(X_train_train, y_train_train, **fit_params)

# Save CV results
pd.DataFrame(grid_search.cv_results_).to_csv(res_folder+'cv_results_gtb.csv')

# Save best params (including n_estimators from early stopping on validation set)
best_params = grid_search.best_estimator_.get_params()
best_params['n_estimators'] = grid_search.best_estimator_.best_ntree_limit
pd.DataFrame(best_params, index=[0]).to_csv(res_folder+'cv_best_params_gtb.csv')


# Gradient boosting regression best model evaluation on test set

# Load best hyper-parameters
best_params = best_params = pd.read_csv(res_folder+'cv_best_params_gtb.csv',
                                        index_col = [0],
                                        usecols = list(params_grid.keys()) + ['n_estimators'])
best_params = best_params.to_dict('records')[0]
best_params['n_jobs'] = 7

# train on whole training set
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train, **fit_params)

# Prediction on test set
y_pred_test['gtb'] = model.predict(X_test) 
y_pred_test.to_hdf(res_folder+'y_pred_test.h5',key='df')


# Model analysis on test set
shap_vals = shap.TreeExplainer(model).shap_values(X_test)
np.save(res_folder + 'shap_values_gtb.npy', shap_vals)




