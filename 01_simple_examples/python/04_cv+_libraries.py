#################################################
# CV + - Regression
# Libraries
#################################################

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from mapie.regression import CrossConformalRegressor

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

# Base model specification
model = RandomForestRegressor(random_state=0)

# Configure CV+ 
mapie = CrossConformalRegressor(
    estimator=model,
    method="plus",        
    cv=10,                
    confidence_level=1 - alpha   
)

# Model training and cross-residual computation
mapie.fit_conformalize(x.reshape(-1, 1), y)

end_time = time.time()

execution_time = end_time - start_time
print(f"MAPIE execution time: {execution_time:.4f} seconds")

# Prediction with intervals for the new test instance
y_pred, y_pis = mapie.predict_interval(np.array([[x_new]]))

# Prediction interval
print("MAPIE Library Interval:")
print(y_pis)
