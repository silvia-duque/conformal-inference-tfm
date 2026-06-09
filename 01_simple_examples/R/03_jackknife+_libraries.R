#################################################
# Jackknife+ - Regression
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

# Format inputs to meet framework constraints
x_adapt <- matrix(x, ncol = 1)
x_new_adapt <- matrix(c(x_new, 3.0), ncol = 1) 

# Define Wrapper Functions for the Base Learner
train_fun_ci <- function(x, y) {
    if (is.null(dim(x))) x <- matrix(x, ncol = 1)
    df <- data.frame(x = x[, 1], y = y)
    set.seed(1)
    randomForest(y ~ x, data = df)
}

predict_fun_ci <- function(out, newx) {
    if (is.null(dim(newx))) newx <- matrix(newx, ncol = 1)
    df <- data.frame(x = newx[, 1])
    predict(out, newdata = df)
}

start_time <- Sys.time()

# Run Jackknife+ 
out_ci <- conformal.pred.jack(
    x = x_adapt,
    y = y,
    x0 = x_new_adapt,
    train.fun = train_fun_ci,
    predict.fun = predict_fun_ci,
    alpha = alpha,
    plus = TRUE
)

end_time <- Sys.time()

execution_time_ci <- as.numeric(end_time - start_time, units = "secs")
print(paste("ConformalInference execution time:", execution_time_ci, "seconds"))

# Extract results corresponding strictly to our target test instance (index 1)
prediction_interval_ci <- c(out_ci$lo[1], out_ci$up[1])
print("ConformalInference Library Interval:")
print(prediction_interval_ci)


#################################################
# Predictset
#################################################

library(predictset)

start_time <- Sys.time()

# Model and Wrapper Specification for Predictset 
rf_model <- make_model(
    train_fun = function(x, y) {
        set.seed(1)
        randomForest(x = data.frame(x = x), y = y)
    },
    predict_fun = function(object, x_new) {
        predict(object, newdata = data.frame(x = x_new))
    },
    type = "regression"
)

# Run Jackknife+
result_ps <- conformal_jackknife(
    x = data.frame(x = x),
    y = y,
    model = rf_model,
    x_new = data.frame(x = x_new),
    alpha = alpha,
    plus = TRUE
)

end_time <- Sys.time()

execution_time_ps <- as.numeric(end_time - start_time, units = "secs")
print(paste("Predictset execution time:", execution_time_ps, "seconds"))

prediction_interval_ps <- c(result_ps$lower, result_ps$upper)
print("Predictset Library Interval:")
print(prediction_interval_ps)