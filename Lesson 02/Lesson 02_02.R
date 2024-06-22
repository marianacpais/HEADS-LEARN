# Compare a decision tree with a random forest in the Cervical Cancer (Risk Factors) data set (available from UCI repository), trying to accurately classify Dx.Cancer

# Load package 'rpart'
library(rpart)
library(caret)

# Load dataset
dataset <- read.csv("./Lesson 02/cervical.csv")

# Transform as factors
col_to_factor <- c(
  "Smokes",
  "Hormonal.Contraceptives",
  "IUD",
  "STDs",
  "STDs.condylomatosis",
  "STDs.cervical.condylomatosis",
  "STDs.vaginal.condylomatosis",
  "STDs.vulvo.perineal.condylomatosis",
  "STDs.syphilis",
  "STDs.pelvic.inflammatory.disease",
  "STDs.genital.herpes",
  "STDs.molluscum.contagiosum",
  "STDs.AIDS",
  "STDs.HIV",
  "STDs.Hepatitis.B",
  "STDs.HPV",
  "STDs..Number.of.diagnosis",
  "STDs..Time.since.first.diagnosis",
  "STDs..Time.since.last.diagnosis",
  "Dx.Cancer",
  "Dx.CIN",
  "Dx.HPV",
  "Dx",
  "Hinselmann",
  "Schiller",
  "Citology",
  "Biopsy"
)

dataset[col_to_factor] <- lapply(dataset[col_to_factor], function(x) {
  x[x == "?"] <- NA
  return(x)
})

dataset[col_to_factor] <- lapply(dataset[col_to_factor], factor)

# Learn tree with Gini impurity
tree.gini <- rpart(Dx.Cancer ~ ., data = dataset)

# Learn tree with information gain
tree.information <- rpart(Dx.Cancer ~ .,
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
fit.cart.rcv <- train(Dx.Cancer ~ ., data = dataset, method = "rpart", metric = metric, trControl = control)

set.seed(7)
fit.rf.rcv <- train(Dx.Cancer ~ ., data = dataset, method = "rf", metric = metric, trControl = control)

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
