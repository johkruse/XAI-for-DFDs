import pandas as pd 
from sklearn import model_selection
import os


# Define area and folder paths
area = 'CE'
folder = './data/{}/'.format(area)

version_folder = folder + 'version-2021-09-08/' #+ pd.Timestamp("today").strftime("%Y-%m-%d") + '/'
if not os.path.exists(version_folder):
   os.makedirs(version_folder)

# We only onclude a subset of features in our model 
# (As we perform an ex-post analysis, we include the actual data from input_actual.h5.
# Additionally, we add day-ahead features not appearing as actual data in input_actual.h5)
additional_day_ahead_features = ['prices_day_ahead', 'month',
                                 'weekday', 'hour', 'price_ramp_day_ahead']


# Load feature data and rocof data
X_actual = pd.read_hdf(folder+'input_actual.h5')
X_forecast = pd.read_hdf(folder + 'input_forecast.h5').loc[:,additional_day_ahead_features]
y = pd.read_hdf(folder + 'outputs.h5').loc[:,['f_rocof']]

# Start and end for continuous time series snippet (for visualization)
start = '2016-12-01 21:00:00' 
end = '2016-12-02 22:00:00' 

# Drop nan values from whole data set
valid_ind =  ~pd.concat([X_forecast, X_actual, y], axis=1).isnull().any(axis=1)
X_forecast, X_actual, y = X_forecast[valid_ind], X_actual[valid_ind], y[valid_ind]

# Join features together
X = X_actual.join(X_forecast)

# Select continuous test set (for visualization)
X_test_cont, y_test_cont = X.loc[start:end].copy(), y.loc[start:end].copy()

# Train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X.drop(index=X_test_cont.index),
                                                                    y.drop(index=X_test_cont.index),
                                                                    test_size=0.2, random_state=42)

# Add continuous time series to test set 
X_test, y_test = X_test.append(X_test_cont), y_test.append(y_test_cont)
y_pred_test = pd.DataFrame(index=y_test.index)


# Save data
X_train.to_hdf(version_folder+'X_train.h5',key='df')
y_train.to_hdf(version_folder+'y_train.h5',key='df')
y_test.to_hdf(version_folder+'y_test.h5',key='df')
y_pred_test.to_hdf(version_folder+'y_pred_test.h5',key='df')
X_test.to_hdf(version_folder+'X_test.h5',key='df')
X.to_hdf(version_folder+'X.h5',key='df')
y.to_hdf(version_folder+'y.h5',key='df')


