#################################################
# Weighted Split Conformal Prediction (WCP) - Regression
# Handling Covariate Shift (From Scratch Implementation)
#################################################

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Data Generation under Covariate Shift
np.random.seed(1)
n = 300

# Training/Calibration distributions drawn from U(0, 5) [P_x]
x = np.random.uniform(0, 5, n)
y = x**2 + np.random.normal(0, 1.5, n)

# Test distribution from U(3, 6) [P_test_x]
test_ref = np.random.uniform(3, 6, (100, 1))

# New Test Instance 
x_new = 4.8
y_true = x_new**2 + np.random.normal(0, 1.5)

# WCP Configuration Parameters
alpha = 0.10

start_time = time.time()

# 1. Split Dataset into Training and Calibration Sets
X_train, X_calib, y_train, y_calib = train_test_split(
    x.reshape(-1, 1), y, test_size=1/3, random_state=1
)

# 2. Train the Base Learner
model_base = RandomForestRegressor(random_state=0)
model_base.fit(X_train, y_train)

# 3. Compute Non-Conformity Scores
pred_cal = model_base.predict(X_calib)
scores_cal = np.abs(y_calib - pred_cal)

# 4. Train a binary classifier to distinguish between Calibration (0) and Test (1)
X_class = np.concatenate([X_calib, test_ref])
y_class = np.concatenate([np.zeros(len(X_calib)), np.ones(len(test_ref))])

# Probabilistic Classifier (Logistic Regression)
prob_model = LogisticRegression()
prob_model.fit(X_class, y_class)

# 5. Compute Weights via Likelihood Ratios using Propensity Scores
def get_w(x_val):
    p_hat = prob_model.predict_proba(x_val)[:, 1]
    p_hat = np.clip(p_hat, 0.01, 0.99) # Numerical stability clipping
    return p_hat / (1 - p_hat)

w_calib = get_w(X_calib)
w_test = get_w(np.array([[x_new]]))

# 6. Normalize Weights integrating the new test instance
sum_w = np.sum(w_calib) + w_test
p_i = w_calib / sum_w
p_test = w_test / sum_w

# 7. Compute the Weighted Conformal Quantile (q_hat_w)
# Sort scores and map their corresponding normalized calibration weights
ord_idx = np.argsort(scores_cal)
s_sorted = scores_cal[ord_idx]
p_sorted = p_i[ord_idx]

# Correct mathematical formulation check
if p_test[0] > alpha:
    q_hat_w = np.inf
else:
    p_cum = np.cumsum(p_sorted)
    idx_q = np.where(p_cum >= (1 - alpha))[0]
    q_hat_w = s_sorted[idx_q[0]]

# 8. Predict and Construct the Weighted Predictive Interval
pred_new = model_base.predict(np.array([[x_new]]))[0]
interval_wcp = [pred_new - q_hat_w, pred_new + q_hat_w]

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("WCP Interval:")
print(interval_wcp)
