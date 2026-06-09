#################################################
# Full Conformal - Regression
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

# Grid of Candidate Values
n_grid_points = 100
y_grid = np.linspace(y.min() - 2, y.max() + 2, n_grid_points)

# Full Conformal Loop
alpha = 0.10
accepted_y = []

start_time = time.time()

for y_candidate in y_grid:
    # 1. Augment the dataset with the candidate value
    x_aug = np.append(x, x_new)
    y_aug = np.append(y, y_candidate)

    # Reshape features for scikit-learn format
    X_aug = x_aug.reshape(-1, 1)

    # 2. Train the base model (e.g., Random Forest)
    model = RandomForestRegressor(random_state=0)
    _ = model.fit(X_aug, y_aug)

    # 3. Generate predictions for the augmented dataset
    pred_aug = model.predict(X_aug)

    # 4. Compute non-conformity scores
    scores = np.abs(y_aug - pred_aug)

    scores_data_prev = scores[:n]
    score_test = scores[n]

    # 5. Compute the conformal quantile
    q_hat = np.quantile(scores_data_prev, 
                        np.ceil((n + 1) * (1 - alpha)) / n)

    # 6. Check decision rule for inclusion
    if score_test <= q_hat:
        accepted_y.append(y_candidate)

end_time = time.time()
print(f"Execution time (seconds): {end_time - start_time:.4f}")

# Continuous prediction interval assumption
continuous_interval = (min(accepted_y), max(accepted_y))
print("Continuous Interval:", continuous_interval)

# Non-contiguous prediction sets (Union of intervals)
accepted_y = np.array(accepted_y)
step = y_grid[1] - y_grid[0]
breaks = np.where(np.diff(accepted_y) > (1.5 * step))[0] + 1
groups = np.split(accepted_y, breaks)
intervals = [(g.min(), g.max()) for g in groups]
print("Prediction Sets (Union of Intervals):", intervals)
