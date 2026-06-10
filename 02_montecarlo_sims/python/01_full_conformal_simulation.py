import os
import time
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

# =================================================================
# Montecarlo Simulation: Full Conformal
# =================================================================


def run_full_simulation(n, sigma, scenario_id, seed_val, alpha):
    np.random.seed(seed_val)

    # 1. Data generation
    x = np.random.uniform(0, 5, n)
    y = x**2 + np.random.normal(0, sigma, n)

    x_new = np.random.uniform(0, 5)
    y_true_obs = x_new**2 + np.random.normal(0, sigma)

    # 2. Full Conformal Execution 
    n_grid_points = 100
    y_grid = np.linspace(y.min() - 2 * sigma, y.max() + 2 * sigma, n_grid_points)

    accepted_y = []

    x_aug = np.append(x, x_new).reshape(-1, 1)
    model = RandomForestRegressor(
        n_estimators=50, n_jobs=1, random_state=seed_val
    )

    k_pos = int(np.ceil((n + 1) * (1 - alpha))) - 1

    # Loop
    for y_candidate in y_grid:
        y_aug = np.append(y, y_candidate)

        model.fit(x_aug, y_aug)

        pred_aug = model.predict(x_aug)
        scores = np.abs(y_aug - pred_aug)

        scores_prev = scores[:n]
        score_test = scores[n]

        q_hat = np.sort(scores_prev)[k_pos]

        if score_test <= q_hat:
            accepted_y.append(y_candidate)

    # 3. Interval Construction
    if len(accepted_y) > 0:
        lower, upper = min(accepted_y), max(accepted_y)
        width = upper - lower
        step = y_grid[1] - y_grid[0]

        accepted_y_sorted = np.sort(accepted_y)
        if len(accepted_y_sorted) > 1:
            is_disjoint = int(np.any(np.diff(accepted_y_sorted) > 1.5 * step))
        else:
            is_disjoint = 0
    else:
        lower, upper, width = np.nan, np.nan, np.nan
        is_disjoint = 0

    # Coverage 
    covered = (
        int(y_true_obs >= lower and y_true_obs <= upper)
        if not np.isnan(lower)
        else 0
    )

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
        "is_disjoint": is_disjoint,
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

        # Parallel execution
        results = Parallel(n_jobs=-2)(
            delayed(run_full_simulation)(
                curr_n, curr_sigma, scenario_id, int(s), ALPHA
            )
            for s in seeds
        )

        all_results.extend(results)

        # Compute scenario execution stats
        scen_time = time.time() - scen_start
        df_scen = pd.DataFrame(results)

        # Summary structure mirroring R data frames
        scen_summary = {
            "scenario_id": scenario_id,
            "marco": "Full",
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

    # Save results to disk
    df_detalle = pd.DataFrame(all_results)
    df_resumen = pd.DataFrame(summary_list)

    df_detalle.to_csv(
        os.path.join(output_dir, "full_detalle_python.csv"),
        index=False,
        sep=";",
    )
    df_resumen.to_csv(
        os.path.join(output_dir, "full_resumen_python.csv"),
        index=False,
        sep=";",
    )

    total_time = time.time() - start_time
    print(
        f"Simulation completed. Total time accumulated: {total_time:.2f} seconds."
    )
