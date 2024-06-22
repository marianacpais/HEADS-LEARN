# IMPORTS
library("e1071")
library("caret")
library("pROC")
dataset <- read.csv("data/dataR2.csv")

dataset$Classification <- as.factor(dataset$Classification)
levels(dataset$Classification) <- make.names(levels(dataset$Classification), unique=TRUE)

# learn SVM with radial kernel (all features)
svm_all <- svm(Classification ~., data=dataset)

summary(svm_all)

plot(
  svm_all,
  data=dataset,
  Glucose ~ Age,
  svSymbol=17,
  dataSymbol=20,
  symbolPalette=c(2,3),
  color.palette=grey.colors,
  slice=list(
    # Age=59.01,
    BMI=27.95,
    # Glucose=98.65,
    Insulin=10.623,
    HOMA=2.9147,
    Adiponectin=10.153,
    Resistin=15.13,
    MCP.1=554.98
  )
)

# RUN ALGORITHMS USING 3 TIMES 10-FOLD CROSS VALIDATION
metric <- "ROC"
control <- trainControl(
  method="repeatedcv",
  number=10,
  summaryFunction=twoClassSummary,
  classProbs=TRUE,
  savePredictions=TRUE,
  repeats=3
)

set.seed(7)
fit_svm_linear <- train(
  Classification ~ ., 
  data=dataset,
  method="svmLinear",
  metric=metric,
  trControl=control,
  preProcess=c("center","scale"),
  tuneLength=10
)
set.seed(7)
fit_svm_radial <- train(
  Classification ~ ., 
  data=dataset,
  method="svmRadial",
  metric=metric,
  trControl=control,
  preProcess=c("center","scale"),
  tuneLength=10
)

# SUMMARIZE ACCURACY OF MODELS
fit_models <- list(fit_svm_linear, fit_svm_radial)
results <- resamples(fit_models)
summary(results)

# ROC curves for models
par(mfrow=c(1,2))
rocs <- lapply(
  fit_models,
  function(fit){
    plot.roc(
      fit$pred$obs,
      fit$pred$X2,
      main=paste("3 x 10-fold CV -", fit$method), 
      debug=FALSE, 
      print.auc=TRUE
    )
  }
)

# COMPARE ACCURACY OF MODELS
dotplot(results)

# INSPECT MODELS
print(fit_svm_linear)
getModelInfo(fit_svm_linear)$svmLinear$parameters

print(fit_svm_radial)
getModelInfo(fit_svm_radial)$svmRadial$parameters

# ROC COMPLEXITY FOR MODELS
plot(fit_svm_radial)

# IMPROVE RADIAL
myGrid <- expand.grid(
  C = c(1,2,4,8),
  sigma=0.09382649
)

set.seed(7)
fit_svm_radial_tune <- train(
  Classification ~ .,
  data=dataset,
  method="svmRadial",
  metric=metric,
  trControl=control,
  tuneGrid=myGrid
)

# ROC COMPLEXITY FOR MODELS
print(fit_svm_radial_tune)

# SUMMARIZE ACCURACY OF MODELS
fit_models <- list(linear=fit_svm_linear, radial=fit_svm_radial, tune=fit_svm_radial_tune)
results <- resamples(fit_models)
summary(results)

# ROC CURVES FOR MODELS
par(mfrow=c(1,2))
rocs <- lapply(
  fit_models,
  function(fit){
    plot.roc(
      fit$pred$obs,
      fit$pred$X2,
      main=paste("3 x 10-fold CV -", fit$method), 
      debug=FALSE, 
      print.auc=TRUE
    )
  }
)

# COMPARE ACCURACY OF MODELS
dotplot(results)



