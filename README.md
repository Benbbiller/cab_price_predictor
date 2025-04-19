# Cab Fare Predictor

This project uses **K-Nearest Neighbors Regression** and **Logistic Regression** to analyze and predict cab ride fares based on time and surge pricing data.

## Models

### KNN Regression
- Predicts actual cab fare using:
  - `hour`, `day_of_week`, `surge_multiplier`
- Evaluation: **MSE**, **RMSE**

### Logistic Regression
- Predicts if a ride is **expensive** (price > $30)
- Uses: `hour`, `surge_multiplier`
- Evaluation: **Accuracy Score**

## Setup and Execution

1. Install dependencies:
  - pip install numpy pandas scikit-learn
2. Place `cab_rides.csv` in your directory.
3. Update the path in the script.
  - Update the path here: 
    - os.chdir('C:/path/')
4. Run it.