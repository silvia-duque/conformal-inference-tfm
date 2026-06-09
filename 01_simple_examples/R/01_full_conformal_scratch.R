#################################################
# Full Conformal - Regression
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

# Grid of Candidate Values
n_grid_points <- 100
y_grid <- seq(min(y) - 2, max(y) + 2, length.out = n_grid_points)

# Full Conformal Loop
accepted_y <- c()
alpha <- 0.10

start_time <- Sys.time()

for (y_candidate in y_grid){
    # 1. Augment the dataset with the candidate value
    x_aug <- c(x, x_new)
    y_aug <- c(y, y_candidate)
    
    # 2. Train the base model (e.g., Random Forest)
    model <- randomForest(
        y ~ x,
        data = data.frame(x = x_aug, y = y_aug)
    )
    
    # 3. Generate predictions for the augmented dataset
    pred_aug <- predict(model, newdata = data.frame(x = x_aug))
    
    # 4. Compute non-conformity scores
    scores <- abs(y_aug - pred_aug)
    
    scores_data_prev <- scores[1:n]
    score_test <- scores[n+1]
    
    # 5. Compute the conformal quantile
    q_hat <- quantile(scores_data_prev, 
                      probs = ceiling((n + 1) * (1 - alpha)) / n)
    
    # 6. Decision rule for inclusion
    if (score_test <= q_hat) {
        accepted_y <- c(accepted_y, y_candidate)
    }
}
end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

# Continuous prediction interval assumption
continuous_interval <- range(accepted_y)
print("Continuous Interval:")
print(continuous_interval)

# Non-contiguous prediction sets (Union of intervals)
step <- diff(y_grid)[1]
groups <- cumsum(c(1, diff(accepted_y) > 1.5 * step))
intervals <- t(sapply(split(accepted_y, groups), range))
print("Prediction Sets (Union of Intervals):")
print(intervals)