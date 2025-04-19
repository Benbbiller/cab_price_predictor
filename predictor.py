import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
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


cab_ridesKnn = KNeighborsRegressor(n_neighbors=100)

cab_ridesKnn.fit(X_train, y_train)

y_pred = cab_ridesKnn.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Knn regression model results: ")
print(f"MSE: ${mse:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"Squared difference between predicted and observed cab fares: {mse:.2f}")
print(f"On average our KNN Regression model cab fare is off by ${rmse:.2f}\n")


#Logistic regression model:

cab_data_refinement['is_expensive'] = cab_data_refinement['price'].apply(lambda x: 1 if x > 30 else 0)

X = cab_data_refinement[['hour', 'surge_multiplier']]
y = cab_data_refinement['is_expensive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_regress = LogisticRegression(penalty='l2')
log_regress.fit(X, y)

print("Logistic Regression model results:\n")

print('w1:', log_regress.coef_)
print('w0:', log_regress.intercept_)

y_pred = log_regress.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(round(score, 3))
print(f"Correctly Predicted {round(score, 3)*100}% of rides as expensive or not expensive\n")
print(f"Between these two models, Logistic regression is more accurate, being able to predict if cab fares are expensive or not {round(score, 3)*100}% of the time. More detailed comparision in the summary.\n")