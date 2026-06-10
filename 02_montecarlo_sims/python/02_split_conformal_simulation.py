import os
import time
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# =================================================================
# Montecarlo Simulation: Split Conformal
# =================================================================


def run_split_simulation(n, sigma, scenario_id, seed_val, alpha):
    np.random.seed(seed_val)

    # 1. Data generation
    x = np.random.uniform(0, 5, n).reshape(-1, 1)
    y = x.flatten() ** 2 + np.random.normal(0, sigma, n)

    x_new = np.random.uniform(0, 5, 1).reshape(-1, 1)
    y_true = x_new.flatten() ** 2 + np.random.normal(0, sigma, 1)
    y_true_obs = y_true[0]

    # 2. Data Splitting 
    X_train, X_cal, y_train, y_cal = train_test_split(
        x, y, test_size=1/2, random_state=seed_val
    )

    # 3. Model training
    model = RandomForestRegressor(
        n_estimators=50, n_jobs=1, random_state=seed_val
    )
    model.fit(X_train, y_train)

    # 4. Non-conformity scores
    pred_cal = model.predict(X_cal)
    scores = np.abs(y_cal - pred_cal)

    # 5. Conformal quantile 
    n_cal = len(scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal

    # q_level = min(q_level, 1.0)
    q_hat = np.quantile(scores, q_level, method="higher")

    # 6. Prediction and Interval construction
    pred_new = model.predict(x_new)[0]
    lower = pred_new - q_hat
    upper = pred_new + q_hat
    width = 2 * q_hat

    # Coverage verification
    covered = int(y_true_obs >= lower and y_true_obs <= upper)

    return {
        "scenario_id": scenario_id,
        "n": n,
        "sigma": sigma,
        "x": x_new[0][0],
        "y_true": y_true_obs,
        "y_hat": pred_new,
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
    N_SIMULATIONS = 19
    ALPHA = 0.05
    np.random.seed(1)

    # Scenarios Definition
    n_values = [50, 100, 200, 1000]
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

        # Parallel execution matching system resources
        results = Parallel(n_jobs=-2)(
            delayed(run_split_simulation)(
                curr_n, curr_sigma, scenario_id, int(s), ALPHA
            )
            for s in seeds
        )

        all_results.extend(results)

        # Compute scenario execution stats
        scen_time = time.time() - scen_start
        df_scen = pd.DataFrame(results)

        # Summary structure mirroring R data frames for easy parsing
        scen_summary = {
            "scenario_id": scenario_id,
            "marco": "Split",
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
        os.path.join(output_dir, "split_detalle_python.csv"),
        index=False,
        sep=";",
    )
    df_resumen.to_csv(
        os.path.join(output_dir, "split_resumen_python.csv"),
        index=False,
        sep=";",
    )

    total_time = time.time() - start_time
    print(
        f"Simulation completed. Total time accumulated: {total_time:.2f} seconds."
    )
