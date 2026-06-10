################################################################################
# Montecarlo Simulation: CV+
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
K_folds <- 10 # Number of folds for CV+

# Scenarios Definition
scenarios <- expand.grid(
    n = c(25, 50, 100, 200, 1000), 
    sigma = c(0.5, 2.0, 4.0)
)

# Parallel initialization
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Set seed
set.seed(1)

# Dataframe to save results
estimates_df <- data.frame()
summary_df   <- data.frame()

start_time <- Sys.time()

for(s in 1:nrow(scenarios)) {
    print(paste("Processing scenario CV+", s, "of", nrow(scenarios)))
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
        
        # 2. CV+ Execution
        folds <- sample(rep(1:K_folds, length.out = curr_n))
        
        cv_residuals <- numeric(curr_n)
        cv_preds_new <- numeric(curr_n)
        
        # Loop across each fold K
        for (k in 1:K_folds) {
            idx_val <- which(folds == k)
            
            # Train model excluding observations from fold k
            model_k <- randomForest(y ~ x, data = data.frame(x = x[-idx_val], y = y[-idx_val]), ntree = 50)
            
            # Predict on excluded observations to compute CV residuals
            pred_val <- predict(model_k, newdata = data.frame(x = x[idx_val]))
            cv_residuals[idx_val] <- abs(y[idx_val] - pred_val)
            
            # Predict model k on the new test instance
            cv_preds_new[idx_val] <- predict(model_k, newdata = data.frame(x = x_new))
        }
        
        # 3. Interval Construction
        lower_vals <- cv_preds_new - cv_residuals
        upper_vals <- cv_preds_new + cv_residuals
        
        q_level_high <- ceiling((curr_n + 1) * (1 - alpha)) / curr_n
        q_level_low  = floor((curr_n + 1) * alpha) / curr_n
        
        lower_bound <- quantile(lower_vals, probs = q_level_low)
        upper_bound <- quantile(upper_vals, probs = q_level_high)
        
        # Results
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
        marco = "CV+",
        n = curr_n,
        sigma = curr_sigma,
        mean_coverage = mean(res_scen$covered),
        mean_width = mean(res_scen$width),
        scen_time_secs = time_scen
    )
    summary_df <- rbind(summary_df, scen_summary)
    
    print(paste("-> ¡Scenario", s, "finished!",
                "Time consumed:", round(time_scen, 2), "seconds"))
}

stopCluster(cl)
end_time <- Sys.time()
total_time <- difftime(end_time, start_time, units = "secs")
print(paste("Simulation completed. Total time accumulated:", round(as.numeric(total_time), 2), "seconds."))

# Save results
saveRDS(estimates_df, file = "simulation_results/cv_plus_detalle.rds")
saveRDS(summary_df, file = "simulation_results/cv_plus_resumen.rds")