#################################################
# Weighted Split Conformal Prediction (WCP) - Regression
# Toy Example with a Single Test Instance
#################################################

library(randomForest)

# Data Generation 
set.seed(1)
n <- 300 

# Training/Calibration distributions drawn from U(0, 5) [P_x]
x <- runif(n, 0, 5)
y <- x^2 + rnorm(n, sd = 1.5)

# Test distribution from U(3, 6) [P_test_x]
test_ref <- data.frame(x = runif(100, 3, 6))

# New Test Instance 
x_new <- 4.8 
y_true <- x_new^2 + rnorm(1, sd = 1.5)

# CCP Configuration Parameters
alpha <- 0.10

start_time <- Sys.time()

# 1. Split Dataset into Training and Calibration Sets
set.seed(1)
idx <- sample(1:n, floor(2/3 * n))
train <- data.frame(x = x[idx], y = y[idx])
calib <- data.frame(x = x[-idx], y = y[-idx])

# 2. Train the Base Learner
model <- randomForest(y ~ x, data = train)

# 3. Compute Non-Conformity Scores (Absolute Residuals)
pred_cal <- predict(model, newdata = calib)
scores_cal <- abs(calib$y - pred_cal)

# 4. Train a binary classifier to distinguish between Calibration (0) and Test (1)

df_class <- rbind(
    data.frame(x = calib$x, target = 0),
    data.frame(x = test_ref$x, target = 1)
)

# Probabilistic Classifier (Logistic Regression)
prob_model <- glm(target ~ x, data = df_class, family = binomial)

# Compute the likelihood ratio w(x) = P(Test|x) / P(Calib|x) using odds ratio
get_w <- function(x_val) {
    p_hat <- predict(prob_model, newdata = data.frame(x = x_val), type = "response")
    p_hat <- pmin(pmax(p_hat, 0.01), 0.99) # Numerical stability clipping
    return(p_hat / (1 - p_hat))
}

w_calib <- get_w(calib$x)
w_test <- get_w(x_new)

# 5. Normalize Weights integrating the new test instance
sum_w <- sum(w_calib) + w_test
p_i <- w_calib / sum_w
p_test <- w_test / sum_w

# 6. Compute the Weighted Conformal Quantile 
ord <- order(scores_cal)
s_sorted <- scores_cal[ord]
p_sorted <- p_i[ord]

p_cum <- cumsum(p_sorted)
idx_q <- which(p_cum >= (1 - alpha))

# Mathematical formulation check
if (p_test > alpha) {
    q_hat_w <- Inf
} else {
    
    q_hat_w <- s_sorted[min(idx_q)]
}

# 7. Weighted  Interval
pred_new <- predict(model, newdata = data.frame(x = x_new))
interval_wcp <- c(pred_new - q_hat_w, pred_new + q_hat_w)

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

print("WCP Interval:")
print(interval_wcp)
