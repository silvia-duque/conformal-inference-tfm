import os
import time
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

# =================================================================
# Montecarlo Simulation: Jackknife+ (Leave-One-Out)
# =================================================================


def run_jackknife_simulation(n, sigma, scenario_id, seed_val, alpha):
    np.random.seed(seed_val)

    # 1. Data generation
    x = np.random.uniform(0, 5, n)
    y = x**2 + np.random.normal(0, sigma, n)

    x_new = np.random.uniform(0, 5)
    y_true_obs = x_new**2 + np.random.normal(0, sigma)

    # Initialize arrays to store Leave-One-Out (LOO) outputs
    loo_residuals = np.zeros(n)
    loo_preds_new = np.zeros(n)

    # 2. Leave-One-Out (LOO) Internal Loop
    for i in range(n):
        # Train the base model on n-1 data points
        X_train = np.delete(x, i).reshape(-1, 1)
        y_train = np.delete(y, i)

        model_i = RandomForestRegressor(
            n_estimators=50, n_jobs=1, random_state=seed_val
        )
        model_i.fit(X_train, y_train)

        # Predict the excluded point to compute the LOO non-conformity score
        pred_i = model_i.predict(np.array([[x[i]]]))[0]
        loo_residuals[i] = abs(y[i] - pred_i)

        # Predict the new test instance using the i-th model
        loo_preds_new[i] = model_i.predict(np.array([[x_new]]))[0]

    # 3. Construct prediction interval components
    lower_vals = loo_preds_new - loo_residuals
    upper_vals = loo_preds_new + loo_residuals

    # 4. Strict Jackknife+ quantile conversions and boundary protection
    q_level_low = np.floor((n + 1) * alpha) / n
    q_level_high = np.ceil((n + 1) * (1 - alpha)) / n

    lower = np.quantile(lower_vals, q_level_low, method="lower")
    upper = np.quantile(upper_vals, q_level_high, method="higher")
    width = upper - lower

    # Coverage
    covered = int(y_true_obs >= lower and y_true_obs <= upper)

    return {
        "scenario_id": scenario_id,
        "n": n,
        "sigma": sigma,
        "x": x_new,
        "y_true": y_true_obs,
        "lower": lower,
        "upper": upper,
        "covered": covered,
        "width": width,
    }


if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = "simulation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    # Simulation Setup
    N_SIMULATIONS = 1900
    ALPHA = 0.05
    np.random.seed(1)

    # Scenarios Definition
    n_values = [25, 50, 100]
    sigma_values = [0.5, 2.0, 4.0]
    scenarios = pd.DataFrame(
        list(product(n_values, sigma_values)), columns=["n", "sigma"]
    )

    all_results = []
    summary_list = []

    for idx, row in scenarios.iterrows():
        scenario_id = int(idx + 1)
        curr_n = int(row["n"])
        curr_sigma = row["sigma"]

        print(
            f"Processing scenario {scenario_id} of {len(scenarios)} | n={curr_n} | sigma={curr_sigma}"
        )

        scen_start = time.time()

        # Generate unique seeds for parallel streams
        seeds = np.random.choice(
            range(1000000), size=N_SIMULATIONS, replace=False
        )

        # Parallel execution distributing montecarlo iterations across threads
        results = Parallel(n_jobs=-2)(
            delayed(run_jackknife_simulation)(
                curr_n, curr_sigma, scenario_id, int(s), ALPHA
            )
            for s in seeds
        )

        all_results.extend(results)

        # Compute scenario execution stats
        scen_time = time.time() - scen_start
        df_scen = pd.DataFrame(results)

        # Summary 
        scen_summary = {
            "scenario_id": scenario_id,
            "marco": "Jackknife+",
            "n": curr_n,
            "sigma": curr_sigma,
            "mean_coverage": df_scen["covered"].mean(),
            "mean_width": df_scen["width"].mean(),
            "scen_time_secs": scen_time,
        }
        summary_list.append(scen_summary)

        print(
            f"-> Scenario {scenario_id} finished! Time consumed: {scen_time:.2f} seconds"
        )

    # Save structured datasets to disk
    df_detalle = pd.DataFrame(all_results)
    df_resumen = pd.DataFrame(summary_list)

    df_detalle.to_csv(
        os.path.join(output_dir, "jackknife_detalle_python.csv"),
        index=False,
        sep=";",
    )
    df_resumen.to_csv(
        os.path.join(output_dir, "jackknife_resumen_python.csv"),
        index=False,
        sep=";",
    )

    total_time = time.time() - start_time
    print(
        f"Simulation completed. Total time accumulated: {total_time:.2f} seconds."
    )
