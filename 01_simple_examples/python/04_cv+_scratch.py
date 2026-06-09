#################################################
# CV+ - Regression
# Toy Example with a Single Test Instance
#################################################

import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor

# Data Generation
np.random.seed(1)
n = 100
x = np.random.uniform(0, 5, n)
y = x**2 + np.random.normal(0, 1.5, n)

# New Test Instance
x_new = 2.5
y_true = x_new**2 + np.random.normal(0, 1.5)

alpha = 0.10

# CV+ Configuration Parameters
K = 10

start_time = time.time()

# 1. Stratified/Random Fold Assignment
folds = np.repeat(np.arange(K), int(np.ceil(n / K)))[:n]
np.random.shuffle(folds)

# Initialize arrays to store Cross-Validation outputs
cv_residuals = np.zeros(n)
cv_preds_new = np.zeros(n)

# 2. Cross-Validation Loop (K Models)
for k in range(K):
    # Extract validation and training indices for the current fold
    idx_val = np.where(folds == k)[0]
    idx_train = np.where(folds != k)[0]
    
    # Define training partition
    X_train = x[idx_train].reshape(-1, 1)
    y_train = y[idx_train]
    
    # Train base learner
    model_k = RandomForestRegressor(random_state=0)
    _ = model_k.fit(X_train, y_train)
    
    # Compute out-of-fold non-conformity scores 
    X_val = x[idx_val].reshape(-1, 1)
    pred_val = model_k.predict(X_val)
    cv_residuals[idx_val] = np.abs(y[idx_val] - pred_val)
    
    # Store predictions for the new test instance from the current fold model
    X_new_arr = np.array([[x_new]])
    cv_preds_new[idx_val] = model_k.predict(X_new_arr)

# 3. Construct Prediction Interval Components
lower_vals = cv_preds_new - cv_residuals
upper_vals = cv_preds_new + cv_residuals

# Apply quantile corrections
q_level_high = np.ceil((n + 1) * (1 - alpha)) / n
q_level_low = np.floor((n + 1) * alpha) / n

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# 4. Predictive Interval
interval_cv_plus = [
    np.quantile(lower_vals, q_level_low),
    np.quantile(upper_vals, q_level_high)
]

print("CV+ Interval:")
print(interval_cv_plus)
