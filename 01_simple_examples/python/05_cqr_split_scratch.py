#################################################
# Conformalized Quantile Regression (CQR) - Split
# Toy Example with a Single Test Instance 
#################################################

import time
import numpy as np
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor

# Data Generation
np.random.seed(1)
n = 100

x = np.random.uniform(0, 5, n)
y = x**2 + np.random.normal(0, 1.5, n)

# New Test Instance
x_new = 2.5
y_true = x_new**2 + np.random.normal(0, 1.5)

alpha = 0.10

# CQR Execution

start_time = time.time()

# 1. Split Dataset into Training and Calibration Sets
X_train, X_calib, y_train, y_calib = train_test_split(
    x.reshape(-1, 1), y, test_size=1/3, random_state=1
)

# 2. Train the Base Quantile Regressor
qrf_model = RandomForestQuantileRegressor(random_state=0)
_ = qrf_model.fit(X_train, y_train)

# 3. Predict Base Quantiles on the Calibration Set
quantiles_target = [alpha / 2, 1 - alpha / 2]
cal_q = qrf_model.predict(X_calib, quantiles=quantiles_target)

# 4. Compute CQR Non-Conformity Scores
scores_cqr = np.maximum(cal_q[:, 0] - y_calib, y_calib - cal_q[:, 1])

# 5. Compute the Conformal Correction Factor (q_hat)
n_cal = len(scores_cqr)
q_hat_cqr = np.quantile(scores_cqr, 
                        np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)

# 6. Predict Base Quantiles for the New Test Instance
test_q = qrf_model.predict(np.array([[x_new]]), quantiles=quantiles_target)

end_time = time.time()

execution_time = end_time - start_time
print(f"Scratch execution time: {execution_time:.4f} seconds")

# 7. Construct the Final Conformalized Interval
cqr_interval = [test_q[0, 0] - q_hat_cqr, test_q[0, 1] + q_hat_cqr]

print("CQR Interval:")
print(cqr_interval)
