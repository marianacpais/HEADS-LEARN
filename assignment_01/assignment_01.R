# Consider the the Cervical Cancer (Risk Factors) data set (available from UCI repository)
# and try to accurately classify Dx.Cancer.

# You must compare different approaches and parameters of
# a) single decision tree and
# b) random forest.

# Evaluation of derived models should follow a correct methodology,
# comparing different estimates of generalization error (i.e. holdout, cross-validation, bootstrap, ...)

# Submit a report (in PDF, generated from R) with the code and the resulting analysis.

library(rpart)
library(caret)


fp <- "assignment_01/cervical.csv"
ds <- read.csv(fp)

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

col_to_numeric <- c(
  "Number.of.sexual.partners",
  "First.sexual.intercourse",
  "Num.of.pregnancies",
  "Smokes..years.",
  "Smokes..packs.year.",
  "Hormonal.Contraceptives..years.",
  "IUD..years.",
  "STDs..number."
)

ds[c(col_to_factor,col_to_numeric)] <- 
  lapply(
    ds[c(col_to_factor,col_to_numeric)], 
    function(x) {
      x[x == "?"] <- NA
      return(x)
    }
  )

ds[col_to_factor] <- lapply(ds[col_to_factor], factor)
ds[col_to_numeric] <- lapply(ds[col_to_numeric], as.numeric)

names(ds) <- c("Age", "NumSexualPartners", "FirstSexualIntercoarse", "NumPregnancies", "Smokes", "SmokesYears", "SmokesPacksYear", "HormonalContraceptives", "HormonalContraceptivesYears", "IUD", "IUDYears", "STDs", "STDsNumber", "STDsCondylomatosis", "STDsCervicalCondylomatosis", "STDsVaginalCondylomatosis", "STDsVulvoPerinealCondylomatosis", "STDsSyphilis", "STDsPelvicInflammatoryDisease", "STDsGenitalHerpes", "STDsMolluscumContagiosum", "STDsAIDS", "STDsHIV", "STDsHepatitisB", "STDsHPV", "STDsNumDiagnosis", "STDsTimeSinceFirstDiagnosis", "STDsTimeSinceLastDiagnosis", "DxCancer", "DxCIN", "DxHPV", "Dx", "Hinselmann", "Schiller", "Citology", "Biopsy")




featurePlot(x=ds[,1:4], y=ds[,5], plot="ellipse")

# Learn tree with Gini impurity
tree.gini <- rpart(Dx.Cancer ~ ., data = ds)

# Learn tree with information gain
tree.information <- rpart(Dx.Cancer ~ .,
  data = ds,
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
fit.cart.rcv <- train(Dx.Cancer ~ ., data = ds, method = "rpart", metric = metric, trControl = control)

set.seed(7)
fit.rf.rcv <- train(Dx.Cancer ~ ., data = ds, method = "rf", metric = metric, trControl = control)

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




#### Single decision tree


#### Random forest