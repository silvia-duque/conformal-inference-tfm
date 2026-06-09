#################################################
# CV+ - Regression
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

alpha <- 0.10

# CV+ Configuration Parameter
K <- 10

start_time <- Sys.time()

# 1. Stratified/Random Fold Assignment
folds <- sample(rep(1:K, length.out = n))

# Initialize vectors to store Cross-Validation outputs
cv_residuals <- numeric(n)
cv_preds_new <- numeric(n)

# 2. Cross-Validation Loop (K Models)
for (k in 1:K) {
    # Extract validation indices for the current fold
    idx_val <- which(folds == k)
    
    # Define training partition excluding current fold data
    X_train <- x[-idx_val]
    y_train <- y[-idx_val]
    
    # Train base learner
    set.seed(1)
    model_k <- randomForest(y ~ x, data = data.frame(x = X_train, y = y_train))
    
    # Compute out-of-fold non-conformity scores 
    pred_val <- predict(model_k, newdata = data.frame(x = x[idx_val]))
    cv_residuals[idx_val] <- abs(y[idx_val] - pred_val)
    
    # Predictions for the new test instance from the current fold model
    cv_preds_new[idx_val] <- predict(model_k, newdata = data.frame(x = x_new))
}

# 3. Construct Prediction Interval Components
lower_vals <- cv_preds_new - cv_residuals
upper_vals <- cv_preds_new + cv_residuals

# Apply quantile corrections
q_level_high <- ceiling((n + 1) * (1 - alpha)) / n
q_level_low  <- floor((n + 1) * alpha) / n

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Scratch execution time:", execution_time, "seconds"))

# 4. Predictive Interval
interval_cv_plus <- c(
    quantile(lower_vals, probs = q_level_low),
    quantile(upper_vals, probs = q_level_high)
)

cat("CV+ Interval:", interval_cv_plus)