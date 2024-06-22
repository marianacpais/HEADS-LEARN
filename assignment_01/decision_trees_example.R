####################################################################################
#
# Decision trees and random forests
#
# by Pedro Pereira Rodrigues <pprodrigues@med.up.pt>
#
####################################################################################

################
### Packages

# Load package 'caret'
library(caret)
# Load package 'ellipse'
library(ellipse)
# Load package 'rpart'
library(rpart)
# Load package 'pROC'
library(pROC)

################
### Data

data(iris)

featurePlot(x=iris[,1:4], y=iris[,5], plot="ellipse")

plot(x=iris$Petal.Length, y=iris$Petal.Width, col=iris$Species, pch=20,
     axes=F, main="Iris Data Set", ylab="Petal Width", xlab="Petal Length")
legend(x=4, y=0.5, cex=.8, legend=levels(iris$Species), col=1:3, pch=20)
axis(1)
axis(2)

tree <- rpart(Species ~ ., data=iris)

lines(x=rep(tree$splits[1,4],2), y=range(iris$Petal.Width), lwd=2, col=4)
lines(y=rep(tree$splits[8,4],2), x=c(tree$splits[1,4], max(iris$Petal.Length)), lwd=2, col=4)

plot(tree)
text(tree)

plot(x=iris$Sepal.Length, y=iris$Sepal.Width, col=iris$Species, pch=20,
     axes=F, main="Iris Data Set", ylab="Sepal Width", xlab="Sepal Length")
legend(x=6.5, y=4.4, cex=.8, legend=levels(iris$Species), col=1:3, pch=20)
axis(1)
axis(2)

tree <- rpart(Species ~ Sepal.Length + Sepal.Width, data=iris)

lines(x=rep(tree$splits[1,4],2), y=range(iris$Sepal.Width), lwd=2, col=4)
lines(y=rep(tree$splits[4,4],2), x=c(min(iris$Sepal.Length),tree$splits[1,4]), lwd=2, col=4)
lines(x=rep(tree$splits[6,4],2), y=range(iris$Sepal.Width), lwd=2, col=4)
lines(y=rep(tree$splits[9,4],2), x=c(tree$splits[1,4],tree$splits[6,4]), lwd=2, col=4)

plot(tree)
text(tree)


################
### Data

# Get data set from UCI repository - Breast Cancer Coimbra
# - available from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

# Define the filename
filename <- "dataR2.csv"

# Load the CSV file from the local directory
dataset <- read.csv(filename, header=T)

# Define Classification as factor
dataset$Classification <- factor(dataset$Classification, levels=1:2, labels=c("Control", "Patient"))

################
# Holdout a validation set, by defining the indices of the training set
set.seed(523)
training.index <- createDataPartition(dataset$Classification, p=0.8, list=FALSE)
validation <- dataset[-training.index,]
dataset <- dataset[training.index,]

# Levels of the class
levels(dataset$Classification)

# Class distribution
proportions <- prop.table(table(dataset$Classification))
cbind(Frequency=table(dataset$Classification), Proportion=round(proportions*100,2))

# Statistical Summary
summary(dataset)

# Learn tree with Gini impurity
tree.gini <- rpart(Classification ~ ., data=dataset)
# Learn tree with information gain
tree.information <- rpart(Classification ~ ., data=dataset,
                          parms = list(split = "information"))

# Summary
summary(tree.information)
summary(tree.gini)

# Plot
par(mfrow=c(1,2))
plot(tree.information, main="Information")
text(tree.information)
plot(tree.gini, main="Gini")
text(tree.gini)


# Run algorithms using 3 times 10-fold cross validation
metric <- "ROC"
control <- trainControl(method="repeatedcv", number=10,
                        summaryFunction=twoClassSummary, 
                        classProbs=T,
                        savePredictions = T, repeats = 3)

set.seed(7)
fit.cart.rcv <- train(Classification ~ ., data=dataset, method="rpart", metric=metric, trControl=control)

set.seed(7)
fit.rf.rcv <- train(Classification ~ ., data=dataset, method="rf", metric=metric, trControl=control)

# Summarize accuracy of models
fit.models <- list(rpart=fit.cart.rcv, rf=fit.rf.rcv)
results <- resamples(fit.models)
summary(results)

# ROC curves for models
par(mfrow=c(1,2))
rocs <- lapply(fit.models, function(fit){plot.roc(fit$pred$obs,fit$pred$Patient,
                                                  main=paste("3 x 10-fold CV -",fit$method), debug=F, print.auc=T)})
# Compare accuracy of models
dotplot(results)

# Inspect models
print(fit.cart.rcv)
#getModelInfo(fit.cart.rcv)
#getModelInfo(fit.cart.rcv)$rpart
getModelInfo(fit.cart.rcv)$rpart$parameters

# Inspect models
print(fit.rf.rcv)
#getModelInfo(fit.rf.rcv)
#getModelInfo(fit.rf.rcv)$rf
getModelInfo(fit.rf.rcv)$rf$parameters

# ROC complexity for models
plot(fit.cart.rcv)
plot(fit.rf.rcv)

# Improve Random Forest
myGrid <- expand.grid(mtry = 1:9)

set.seed(7)
fit.rf.rcv.tune <- train(Classification ~ ., data=dataset, method="rf", metric=metric, trControl=control,
                         tuneGrid = myGrid)

# ROC complexity for models
print(fit.rf.rcv.tune)
plot(fit.rf.rcv.tune)

# Summarize accuracy of models
fit.models <- list(rpart=fit.cart.rcv, rf=fit.rf.rcv, rf.tune=fit.rf.rcv.tune)
results <- resamples(fit.models)
summary(results)

# ROC curves for models
par(mfrow=c(1,3))
rocs <- lapply(fit.models, function(fit){plot.roc(fit$pred$obs,fit$pred$Patient,
                                                  main=paste("3 x 10-fold CV -",fit$method), debug=F, print.auc=T)})
# Compare accuracy of models
dotplot(results)

################
# Make predictions

par(mfrow=c(1,1))

# Estimate skill of RF on the validation dataset
predictions.prob <- predict(fit.rf.rcv.tune, validation, type="prob")
predictions <- predict(fit.rf.rcv.tune, validation, type="raw")
confusionMatrix(predictions, validation$Classification)
plot.roc(validation$Classification, predictions.prob$Patient, print.auc=T, axes=F, main=paste("3 x 10-fold CV -",fit.rf.rcv.tune$method), debug=F)
axis(1)
axis(2)

