#################################################
# CV+ - Regression
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
result <- conformal_cv(
    x = data.frame(x = x),
    y = y,
    model = rf_model,
    x_new = data.frame(x = x_new),
    alpha = 0.1,
    n_folds = 10
)

end_time <- Sys.time()

execution_time_ps <- as.numeric(end_time - start_time, units = "secs")
print(paste("Predictset execution time:", execution_time_ps, "seconds"))

prediction_interval_ps <- c(result$lower, result$upper)
print("Predictset Library Interval:")
print(prediction_interval_ps)