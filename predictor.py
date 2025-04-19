import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

# k-nearest neighbor regression model:

            #replace with your directory where csv is stored 
os.chdir('C:/Users/ecoin/478_project/cab_price_predictor') 

cab_rides = pd.read_csv('cab_rides.csv', nrows=60000).dropna()
cab_data_refinement = cab_rides.drop(columns=['id', 'product_id'])

cab_data_refinement['datetime'] = pd.to_datetime(cab_data_refinement['time_stamp'] ,unit='ms')
cab_data_refinement['hour'] = cab_data_refinement['datetime'].dt.hour
cab_data_refinement['day_of_week'] = cab_data_refinement['datetime'].dt.dayofweek

X = cab_data_refinement[['hour','day_of_week', 'surge_multiplier']]

y = cab_data_refinement['price']

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


cab_ridesKnn = KNeighborsRegressor(n_neighbors=3)

cab_ridesKnn.fit(X_train, y_train)

y_pred = cab_ridesKnn.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
#score = accuracy_score(y_test, y_pred)

print(f"MAE: ${mse:.2f}")
print(f"RMSE: ${rmse:.2f}")

#print('Accuracy score is ', end="")
#print('%.3f' % score)