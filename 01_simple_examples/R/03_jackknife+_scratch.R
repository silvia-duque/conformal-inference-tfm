#################################################
# Jackknife+ - Regression
# Toy Example with a Single Test Instance
#################################################

library(randomForest)

# Data Generation
set.seed(1)
n <- 100  
x <- runif(n, 0, 5)
y <- x^2 + rnorm(n, sd = 1.5)

# New Test Instance 
x_new <- 2.5
y_true <- x_new^2 + rnorm(1, sd = 1.5)

# Split Conformal Execution
alpha <- 0.10

start_time <- Sys.time()

# Initialize vectors to store Leave-One-Out (LOO) outputs
loo_residuals <- numeric(n)
loo_preds_new <- numeric(n)

# Leave-One-Out (LOO) Loop
for (i in 1:n) {
    # 1. Train the base model on n-1 data points
    model_i <- randomForest(y ~ x, data = data.frame(x = x[-i], y = y[-i]))
    
    # 2. Predict the excluded point to compute the LOO non-conformity score
    pred_i <- predict(model_i, newdata = data.frame(x = x[i]))
    loo_residuals[i] <- abs(y[i] - pred_i)
    
    # 3. Predict the new test instance using the i-th model
    loo_preds_new[i] <- predict(model_i, newdata = data.frame(x = x_new))
}

# 4. Construct the prediction interval components
lower_vals <- loo_preds_new - loo_residuals
upper_vals <- loo_preds_new + loo_residuals

# Apply the quantile corrections
q_level_high <- ceiling((n + 1) * (1 - alpha)) / n
q_level_low  <- floor((n + 1) * alpha) / n

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Scratch execution time:", execution_time, "seconds"))

# 5. Construct the final predictive interval
interval_jk_plus <- c(
    quantile(lower_vals, probs = q_level_low), 
    quantile(upper_vals, probs = q_level_high)
)
cat("Jackknife+ Prediction Interval:", interval_jk_plus)
