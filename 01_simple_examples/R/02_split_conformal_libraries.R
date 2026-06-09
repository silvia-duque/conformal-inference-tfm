#################################################
# Split Conformal - Regression
# Libraries
#################################################

# Data Generation
set.seed(1)
n <- 100  
x <- runif(n, 0, 5)
y <- x^2 + rnorm(n, sd = 1.5)

# New Test Instance 
x_new <- 2.5
y_true <- x_new^2 + rnorm(1, sd = 1.5)

alpha <- 0.10


#################################################
# ConformalInference
#################################################

library(conformalInference)
library(randomForest)

# Define Wrapper Functions for the Base Learner
train.fun <- function(x, y) {
    df <- data.frame(x = x[, 1], y = y)
    randomForest(y ~ x, data = df)
}

predict.fun <- function(out, newx) {
    df <- data.frame(x = newx[, 1])
    predict(out, newdata = df)
}

# Run Split Conformal Prediction using the framework
start_time <- Sys.time()

out <- conformal.pred.split(
    x = x,
    y = y,
    x0 = 2.5,
    train.fun = train.fun,
    predict.fun = predict.fun,
    alpha = alpha,
    rho = 2/3
)

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

prediction_interval <- c(out$lo, out$up)
print("ConformalInference Library Interval:")
print(continuous_interval)

#################################################
# Probably
#################################################

library(tidymodels)
library(probably)

start_time <- Sys.time()

# Split dataset into training and calibration sets
idx <- sample(1:n, floor(2/3 * n))

train <- data.frame(x = x[idx], y = y[idx])
cal <- data.frame(x = x[-idx], y = y[-idx])

# Random forest model (tidymodels)
rf_spec <- rand_forest() %>%
    set_engine("ranger") %>%
    set_mode("regression")

# Tidymodels workflow
wf <- workflow() %>%
    add_formula(y ~ x) %>%
    add_model(rf_spec)

fit_model <- fit(wf, data = train)

# Split conformal
conf <- int_conformal_split(
    fit_model,
    cal
)

end_time <- Sys.time()
execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

# Prediction + interval
print("Probably Library Interval:")
predict(conf, new_data = data.frame(x = x_new), level = 1 - alpha)

#################################################
# Predictset
#################################################

library(predictset)

start_time <- Sys.time()

# Model and wrapper functions specification
rf_model <- make_model(
    train_fun = function(x, y) {
        randomForest(x = data.frame(x = x), y = y)
    },
    predict_fun = function(object, x_new) {
        predict(object, newdata = data.frame(x = x_new))
    },
    type = "regression"
)

# Calibrate and compute the split conformal prediction
result <- conformal_split(
    x = data.frame(x = x),
    y = y,
    model = rf_model,
    x_new = data.frame(x = x_new),
    alpha = alpha,
    cal_fraction = 0.3
)

end_time <- Sys.time()
execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

# Interval
interval <- c(result$lower, result$upper)
print("Predictset Library Interval:")
print(interval)
