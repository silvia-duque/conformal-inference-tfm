#################################################
# Jackknife+ - Regression
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

# Jackknife+ Execution

start_time = time.time()

# Initialize arrays to store Leave-One-Out (LOO) outputs
loo_residuals = np.zeros(n)
loo_preds_new = np.zeros(n)

# Leave-One-Out (LOO) Loop
for i in range(n):
    # 1. Train the base model on n-1 data points
    X_train = np.delete(x, i).reshape(-1, 1)
    y_train = np.delete(y, i)

    model_i = RandomForestRegressor(random_state=0)
    _ = model_i.fit(X_train, y_train)
    
    # 2. Predict the excluded point to compute the LOO non-conformity score
    pred_i = model_i.predict(np.array([[x[i]]]))[0]
    loo_residuals[i] = abs(y[i] - pred_i)
    
    # 3. Predict the new test instance using the i-th model
    loo_preds_new[i] = model_i.predict(np.array([[x_new]]))[0]

# 4. Construct the prediction interval components
lower_vals = loo_preds_new - loo_residuals
upper_vals = loo_preds_new + loo_residuals

# Apply the strict Jackknife+ quantile corrections
q_level_low = np.floor((n + 1) * alpha) / n
q_level_high = np.ceil((n + 1) * (1 - alpha)) / n

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# 5. Construct the final predictive interval
interval_jk_plus = (
    np.quantile(lower_vals, q_level_low),
    np.quantile(upper_vals, q_level_high)
)
print("Jackknife+ Interval:")
print(interval_jk_plus)
