####################################################################################
#
# Your first Machine Learning Project in R - Step by Step
#
# - based on https://machinelearningmastery.com/machine-learning-in-r-step-by-step/
#
# by Pedro Pereira Rodrigues <pprodrigues@med.up.pt>
#
####################################################################################

################
### Packages

# Load package 'caret'
library(caret)

################
### Data

# Get data set from UCI repository - Breast Cancer Coimbra
# - available from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

# Define the filename
filename <- "data/dataR2.csv"

# Load the CSV file from the local directory
dataset <- read.csv(filename, header=T)

# Define Classification as factor
dataset$Classification <- factor(dataset$Classification, levels=1:2, labels=c("Control", "Patient"))

################
# Holdout a validation set, by defining the indices of the training set
training.index <- createDataPartition(dataset$Classification, p=0.8, list=FALSE)
validation <- dataset[-training.index,]
dataset <- dataset[training.index,]

################
### Summarize

# Dimensions (should be 116 observations and 10 variables)
dim(dataset)

# Types of variables
sapply(dataset, class)

# Take a peek at the data
head(dataset)

# Levels of the class
levels(dataset$Classification)

# Class distribution
proportions <- prop.table(table(dataset$Classification))
cbind(Frequency=table(dataset$Classification), Proportion=round(proportions*100,2))

# Statistical Summary
summary(dataset)

################
# Visualize

# Split input and output
input <- dataset[,-10]
output <- dataset[,10]

# Boxplot for each variable
par(mfrow=c(3,3))
bplots <- lapply(input, boxplot)

# Barplot for class breakdown
par(mfrow=c(1,1))
plot(output)

# Multivariate plot
featurePlot(x=input, y=output, plot="ellipse")

# Box and whisker plots
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=input, y=output, plot="box", scales=scales)

# Density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=input, y=output, plot="density", scales=scales)


################
# Test harness

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# A) Linear algorithms
# Linear Discriminant Analysis
set.seed(7)
fit.lda <- train(Classification ~ ., data=dataset, method="lda", metric=metric, trControl=control)

# B) Non-Linear algorithms
# CART
set.seed(7)
fit.cart <- train(Classification ~ ., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Classification ~ ., data=dataset, method="knn", metric=metric, trControl=control)

# C) Advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Classification ~ ., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Classification ~ ., data=dataset, method="rf", metric=metric, trControl=control)

# Summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# Compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.svm)

################
# Make predictions

# Estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, validation)
confusionMatrix(predictions, validation$Classification)
