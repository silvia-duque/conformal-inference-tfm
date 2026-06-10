################################################################################
# Montecarlo Simulation: Full Conformal 
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
n_grid_points <- 100 

# Scenarios Definition
scenarios <- expand.grid(
    n = c(25, 50, 100), 
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
        
        # 2. Full Conformal Execution
        y_grid <- seq(min(y) - 2 * curr_sigma, max(y) + 2 * curr_sigma, 
                      length.out = n_grid_points)
        accepted_y <- c()
        
        for (y_candidate in y_grid) {
            x_aug <- c(x, x_new)
            y_aug <- c(y, y_candidate)
            
            model <- randomForest(y ~ x, data = data.frame(x = x_aug, y = y_aug), ntree = 50)
            
            pred_aug <- predict(model, newdata = data.frame(x = x_aug))
            scores <- abs(y_aug - pred_aug)
            
            scores_prev <- scores[1:curr_n]
            score_test <- scores[curr_n + 1]
            
            q_hat <- quantile(scores_prev, probs = ceiling((curr_n + 1) * (1 - alpha)) / curr_n)
            
            if (score_test <= q_hat) {
                accepted_y <- c(accepted_y, y_candidate)
            }
        }
        
        if(length(accepted_y) > 0) {
            lower_bound <- min(accepted_y)
            upper_bound <- max(accepted_y)
            step <- diff(y_grid)[1]
            is_disjoint <- any(diff(accepted_y) > 1.5 * step)
        } else {
            lower_bound <- NA; upper_bound <- NA
        }
        
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
            width = upper_bound - lower_bound,
            is_disjoint = is_disjoint
        )
    }
    
    end_scen <- Sys.time()
    time_scen <- as.numeric(difftime(end_scen, start_scen, units = "secs"))
    
    estimates_df <- rbind(estimates_df, res_scen)
    
    # Summary 
    scen_summary <- data.frame(
        scenario_id = s,
        marco = "Full",
        n = curr_n,
        sigma = curr_sigma,
        mean_coverage = mean(res_scen$covered, na.rm = TRUE),
        mean_width = mean(res_scen$width, na.rm = TRUE),
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
saveRDS(estimates_df, file = "simulation_results/full_detalle.rds")
saveRDS(summary_df, file = "simulation_results/full_resumen.rds")