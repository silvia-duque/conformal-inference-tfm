#################################################
# Conformalized Quantile Regression (CQR) - Split
# Toy Example with a Single Test Instance
#################################################

library(quantregForest)

# Data Generation
set.seed(1)
n <- 100  
x <- runif(n, 0, 5)
y <- x^2 + rnorm(n, sd = 1.5)

# New Test Instance 
x_new <- 2.5
y_true <- x_new^2 + rnorm(1, sd = 1.5)

alpha <- 0.10

# CQR Execution
start_time <- Sys.time()

# 1. Split Dataset into Training and Calibration Sets
idx <- sample(1:n, floor(2/3 * n))
train <- data.frame(x = x[idx], y = y[idx])
calib <- data.frame(x = x[-idx], y = y[-idx])

# 2. Train the Base Quantile Regressor
set.seed(1) 
qrf_model <- quantregForest(x = as.matrix(train$x), y = train$y)

# 3. Predict Base Quantiles on the Calibration Set
cal_q <- predict(qrf_model, newdata = as.matrix(calib$x), what = c(alpha / 2, 1 - alpha / 2))

# 4. Compute CQR Non-Conformity Scores
scores_cqr <- pmax(cal_q[, 1] - calib$y, calib$y - cal_q[, 2])

# 5. Compute the Conformal Quantile
n_cal <- length(scores_cqr)
q_hat_cqr <- quantile(scores_cqr, probs = ceiling((n_cal + 1) * (1 - alpha)) / n_cal)

# 6. Predict Base Quantiles for the New Test Instance
test_q <- predict(qrf_model, newdata = as.matrix(x_new), what = c(alpha / 2, 1 - alpha / 2))

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Scratch execution time:", execution_time, "seconds"))

# 7. Conformalized Interval
cqr_interval <- c(test_q[, 1] - q_hat_cqr, test_q[, 2] + q_hat_cqr)

print("CQR Interval:")
print(cqr_interval)