# Load package 'rpart'
library(rpart)
library(caret)

# Load dataset
dataset <- read.csv("./Lesson 02/dataR2.csv")

# Learn tree with Gini impurity
tree.gini <- rpart(Classification ~ ., data = dataset)

# Learn tree with information gain
tree.information <- rpart(Classification ~ .,
  data = dataset,
  parms = list(split = "information")
)

# Summary
summary(tree.information)
summary(tree.gini)

# Plot
par(mfrow = c(1, 2))
plot(tree.information, main = "Information")
text(tree.information)
plot(tree.gini, main = "Gini")
text(tree.gini)

##########

# Run algorithms using 3 times 10-fold cross validation
metric <- "ROC"
control <- trainControl(
  method = "repeatedcv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = T,
  savePredictions = T, repeats = 3
)

set.seed(7)
fit.cart.rcv <- train(Classification ~ ., data = dataset, method = "rpart", metric = metric, trControl = control)

set.seed(7)
fit.rf.rcv <- train(Classification ~ ., data = dataset, method = "rf", metric = metric, trControl = control)

# Summarize accuracy of models
fit.models <- list(cart = fit.cart.rcv, rf = fit.rf.rcv)
results <- resamples(fit.models)
summary(results)

# ROC curves for models
par(mfrow = c(1, 2))
rocs <- lapply(fit.models, function(fit) {
  plot.roc(fit$pred$obs, fit$pred$Patient,
    main = paste("3 x 10-fold CV -", fit$method), debug = F, print.auc = T
  )
})

# Compare accuracy of models
dotplot(results)
