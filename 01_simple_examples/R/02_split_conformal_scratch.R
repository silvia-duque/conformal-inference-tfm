#################################################
# Split Conformal - Regression
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

# 1. Split the dataset into training and calibration sets (2/3 train, 1/3 calib)
idx <- sample(1:n, floor(2/3 * n))
train <- data.frame(x = x[idx], y = y[idx])
calib <- data.frame(x = x[-idx], y = y[-idx])

# 2. Train the base model on the training set
model <- randomForest(y ~ x, data = train)

# 3. Generate predictions for the calibration dataset
pred_cal <- predict(model, newdata = calib)

# 4. Compute non-conformity scores
scores <- abs(calib$y - pred_cal)

# 5. Compute the conformal quantile
n_cal <- length(scores)
q_hat <- quantile(scores, probs = ceiling((n_cal + 1) * (1 - alpha)) / n_cal)

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

# 6. Generate prediction for the new test instance
pred_new <- predict(model, newdata = data.frame(x = x_new))

# 7. Construct the prediction interval
continuous_interval <- c(pred_new - q_hat, pred_new + q_hat)
print("Continuous Interval:")
print(continuous_interval)
