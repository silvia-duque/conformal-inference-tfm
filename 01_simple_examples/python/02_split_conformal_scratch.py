#################################################
# Split Conformal - Regression
# Toy Example with a Single Test Instance
#################################################

import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Data Generation
np.random.seed(1)
n = 100
x = np.random.uniform(0, 5, n)
y = x**2 + np.random.normal(0, 1.5, n)

# New Test Instance
x_new = 2.5
y_true = x_new**2 + np.random.normal(0, 1.5)

# Split Conformal Execution
alpha = 0.10

start_time = time.time()

# 1. Split the dataset into training and calibration sets (2/3 train, 1/3 calib)
x_train, x_cal, y_train, y_cal = train_test_split(
    x.reshape(-1, 1), y, test_size=1/3, random_state=1
)

# 2. Train the base model on the training set
model = RandomForestRegressor(random_state=0)
model.fit(x_train, y_train)

# 3. Generate predictions for the calibration dataset
pred_cal = model.predict(x_cal)

# 4. Compute non-conformity scores
scores = np.abs(y_cal - pred_cal)

# 5. Compute the conformal quantile
n_cal = len(scores)
q_hat = np.quantile(scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# 6. Generate prediction for the new test instance
pred_new = model.predict([[x_new]])[0]

# 7. Construct the prediction interval
interval = [pred_new - q_hat, pred_new + q_hat]
print("Prediction Interval:")
print(interval)
