#################################################
# Full Conformal - Regression
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


#################################################
# ConformalInference
#################################################

library(conformalInference)
library(randomForest)

# Define Wrapper Functions for the Base Learner
train_fun <- function(x, y, ...) {
    df <- data.frame(x = x)
    randomForest(y ~ x, data = df)
}

predict_fun <- function(obj, newx) {
    df <- data.frame(x = newx)
    predict(obj, newdata = df)
}

# Run Full Conformal Prediction using the framework
start_time <- Sys.time()

out <- conformal.pred(
    x = x,
    y = y,
    x0 = x_new,
    train.fun = train_fun,
    predict.fun = predict_fun,
    alpha = 0.1,
    num.grid.pts = 100,
    grid.factor = 1.25
)

end_time <- Sys.time()

execution_time <- as.numeric(end_time - start_time, units = "secs")
print(paste("Execution time:", execution_time, "seconds"))

print("Conformal Inference Library Interval:")
print(paste("Lower bound (lo):", out$lo))
print(paste("Upper bound (up):", out$up))