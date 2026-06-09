#################################################
# Split Conformal - Regression
# Libraries
#################################################

import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mapie.regression import SplitConformalRegressor

# Data Generation
np.random.seed(1)
n = 100
x = np.random.uniform(0, 5, n)
y = x**2 + np.random.normal(0, 1.5, n)

# New Test Instance
x_new = 2.5
y_true = x_new**2 + np.random.normal(0, 1.5)

alpha = 0.10


#################################################
# MAPIE
#################################################

start_time = time.time()

# Split dataset into training and calibration sets
X_train, X_cal, y_train, y_cal = train_test_split(
    x.reshape(-1, 1), y, test_size=1/3, random_state=1
)

# Base model specification
model = RandomForestRegressor(random_state=0)

# Configure the split conformal regressor 
mapie = SplitConformalRegressor(
    estimator=model,
    prefit=False,
    confidence_level=1 - alpha
)

# Model training
mapie.fit(X_train, y_train)

# Calibration step
mapie.conformalize(X_cal, y_cal)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Prediction and interval generation for the new test instance
y_pred, y_pis = mapie.predict_interval([[x_new]])

# Extract and format the prediction interval
print("MAPIE Prediction Interval:")
print(y_pis)
