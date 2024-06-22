# Consider the the Cervical Cancer (Risk Factors) data set (available from UCI 
# repository) and try to accurately classify Dx.Cancer.

# You must compare different approaches and parameters of support vector 
# machines.
 
# Evaluation of derived models should follow a correct methodology, 
# comparing different estimates of generalization error 
#(i.e. holdout, cross-validation, bootstrap, ...)

# Submit a report (in PDF, generated from R) with the code and the resulting 
# analysis.

# IMPORTS
library("e1071")
library("caret")
library("pROC")
library("ROSE")

# DATA TRANSFORMATIONS
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

names(ds) <- c(
  "Age", 
  "NumSexualPartners", 
  "FirstSexualIntercourse", 
  "NumPregnancies", 
  "Smokes", 
  "SmokesYears", 
  "SmokesPacksYear", 
  "HormonalContraceptives", 
  "HormonalContraceptivesYears", 
  "IUD", 
  "IUDYears", 
  "STDs", 
  "STDsNumber", 
  "STDsCondylomatosis", 
  "STDsCervicalCondylomatosis", 
  "STDsVaginalCondylomatosis", 
  "STDsVulvoPerinealCondylomatosis", 
  "STDsSyphilis", 
  "STDsPelvicInflammatoryDisease", 
  "STDsGenitalHerpes", 
  "STDsMolluscumContagiosum", 
  "STDsAIDS", 
  "STDsHIV", 
  "STDsHepatitisB", 
  "STDsHPV", 
  "STDsNumDiagnosis", 
  "STDsTimeSinceFirstDiagnosis", 
  "STDsTimeSinceLastDiagnosis", 
  "DxCancer", 
  "DxCIN", 
  "DxHPV", 
  "Dx", 
  "Hinselmann", 
  "Schiller", 
  "Citology", 
  "Biopsy"
)

# Excluding specific variables
ds_svm <- ds[, !(names(ds) %in% c("SmokesYears", "SmokesPacksYear", "HormonalContraceptives", 
                                  "IUD", "STDs", "STDsNumber", "STDsNumDiagnosis", 
                                  "STDsCervicalCondylomatosis", "STDsAIDS", "DxCIN", 
                                  "DxHPV", "Hinselmann", "Schiller", "Citology", "Biopsy",
                                  "STDsTimeSinceFirstDiagnosis", "STDsTimeSinceLastDiagnosis"))]

# Update Dx to factor with proper naming
ds_svm$Dx <- factor(ds_svm$Dx, levels = c("0", "1"), labels = c("No", "Yes"))

# Splitting Data into Training and Testing Sets
set.seed(7)
training_indices <- createDataPartition(ds_svm$Dx, p = 0.8, list = FALSE)
training_data <- ds_svm[training_indices, ]
testing_data <- ds_svm[-training_indices, ]

# Remove rows with any NA values from the training data
training_data_cleaned <- na.omit(training_data)

# Using ROSE to balance the classes in the cleaned training data
set.seed(123)
training_data_cleaned <- ovun.sample(Dx ~ ., data = training_data_cleaned, method = "both", p = 0.5, seed = 123, N = 1600)$data

# Check the new distribution of the target variable after balancing
table(training_data_cleaned$Dx)

# Setup for cross-validation
control <- trainControl(method="repeatedcv",
                        number=10,
                        summaryFunction=twoClassSummary,
                        classProbs=TRUE,
                        savePredictions=TRUE,
                        repeats=3)

# Train SVM with Linear Kernel on balanced data
set.seed(7)
svm_linear <- train(Dx ~ ., data = training_data_balanced, 
                    method="svmLinear", 
                    metric="ROC", 
                    trControl=control,
                    tuneLength=10)

# Train SVM with Radial Kernel on balanced data
set.seed(7)
svm_radial <- train(Dx ~ ., data = training_data_balanced, 
                    method="svmRadial", 
                    metric="ROC", 
                    trControl=control,
                    tuneLength=10)

# Summarize and compare model performance
fit_models <- list(Linear = svm_linear, Radial = svm_radial)
results <- resamples(fit_models)
print(summary(results))

# Plotting ROC Curves for both models
roc_data_linear <- roc(response=svm_linear$pred$obs, predictor=as.numeric(svm_linear$pred$Yes))
roc_data_radial <- roc(response=svm_radial$pred$obs, predictor=as.numeric(svm_radial$pred$Yes))

# ROC plot for Linear SVM
plot(roc_data_linear, main="ROC Curve for Linear SVM", col="blue")
# Add ROC plot for Radial SVM
plot(roc_data_radial, add=TRUE, col="red")
legend("bottomright", legend=c("Linear SVM", "Radial SVM"), col=c("blue", "red"), lwd=2)

dotplot(results)

# Inspect Models
print(svm_linear)
getModelInfo("svmLinear")$svmLinear$parameters

print(svm_radial)
getModelInfo("svmRadial")$svmRadial$parameters

# Tuning the radial SVM model
myGrid <- expand.grid(C = c(1, 2, 4, 8), sigma = 0.09382649)

set.seed(7)
fit_svm_radial_tune <- train(Dx ~ ., data = training_data_balanced, 
                             method = "svmRadial",
                             metric = "ROC",
                             trControl = control,
                             tuneGrid = myGrid)

# Summarize accuracy of models including the tuned radial model
fit_models <- list(Linear = svm_linear, Radial = svm_radial, TunedRadial = fit_svm_radial_tune)
results <- resamples(fit_models)
print(summary(results))

dotplot(results)
