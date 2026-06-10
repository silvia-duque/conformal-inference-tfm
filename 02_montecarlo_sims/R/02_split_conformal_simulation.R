################################################################################
# Montecarlo Simulation: Split Conformal
################################################################################

library(randomForest)
library(foreach)
library(doParallel)

# Ensure output directory exists
if (!dir.exists("simulation_results")) {
    dir.create("simulation_results")
}

# Simulation Setup
n_sim <- 1900 # Number of simulations to guarantee a MC SE <= 0.5%
alpha <- 0.05 # 95% nominal coverage

# Scenarios Definition
scenarios <- expand.grid(
    n = c(50, 100, 200, 1000), 
    sigma = c(0.5, 2.0, 4.0)
)

# Parallel initialization
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Set seed
set.seed(1)

# Initialize dataframes to save results
estimates_df <- data.frame()
summary_df   <- data.frame()

start_time <- Sys.time()

for(s in 1:nrow(scenarios)) {
    print(paste("Processing scenario", s, "of", nrow(scenarios)))
    curr_n <- scenarios$n[s]
    curr_sigma <- scenarios$sigma[s]
    
    start_scen <- Sys.time()
    
    res_scen <- foreach(iter = 1:n_sim, .combine = rbind, .packages = 'randomForest') %dopar% {
        # 1. Data generation
        x <- runif(curr_n, 0, 5)
        y <- x^2 + rnorm(curr_n, sd = curr_sigma)
        
        # New instance for testing
        x_new <- runif(1, 0, 5)
        y_true <- x_new^2 + rnorm(1, sd = curr_sigma)
        
        # 2. Split Conformal Execution
        idx <- sample(1:curr_n, floor(1/2 * curr_n))
        train <- data.frame(x = x[idx], y = y[idx])
        calib <- data.frame(x = x[-idx], y = y[-idx])
        
        model <- randomForest(y ~ x, data = train, ntree = 50)
        
        pred_cal <- predict(model, newdata = calib)
        scores <- abs(calib$y - pred_cal)
        
        n_cal <- length(scores)
        q_hat <- quantile(scores, probs = ceiling((n_cal + 1) * (1 - alpha)) / n_cal)
        
        pred_new <- predict(model, newdata = data.frame(x = x_new))
        
        lower_bound <- pred_new - q_hat
        upper_bound <- pred_new + q_hat
        
        # 3. Results
        data.frame(
            scenario_id = s,
            n = curr_n,
            sigma = curr_sigma,
            x = x_new,
            y_true = y_true,
            lower = lower_bound,
            upper = upper_bound,
            covered = as.numeric(y_true >= lower_bound && y_true <= upper_bound),
            width = upper_bound - lower_bound
        )
    }
    
    end_scen <- Sys.time()
    time_scen <- as.numeric(difftime(end_scen, start_scen, units = "secs"))
    
    estimates_df <- rbind(estimates_df, res_scen)
    
    # Summary
    scen_summary <- data.frame(
        scenario_id = s,
        marco = "Split",
        n = curr_n,
        sigma = curr_sigma,
        mean_coverage = mean(res_scen$covered),
        mean_width = mean(res_scen$width),
        scen_time_secs = time_scen
    )
    summary_df <- rbind(summary_df, scen_summary)
    
    print(paste("-> ¡Scenario", s, "finished!",
                "Time consumed:", round(time_scen, 2), "seconds."))
}

stopCluster(cl)
end_time <- Sys.time()
total_time <- difftime(end_time, start_time, units = "secs")
print(paste("Simulation completed. Total time accumulated:", round(as.numeric(total_time), 2), "seconds."))

# Save results
saveRDS(estimates_df, file = "simulation_results/split_detalle.rds")
saveRDS(summary_df, file = "simulation_results/split_resumen.rds")