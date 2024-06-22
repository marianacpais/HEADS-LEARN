# IMPORTS
library(e1071)
iris <- data(iris)

# LINEAR - 2 FEATURES

svm_iris_linear <- svm(Species ~ Petal.Width + Petal.Length, data=iris, kernel="linear")
summary(svm_iris_linear)

pred_linear <- predict(svm_iris_linear, x)
print(table(pred_linear, y))

plot(
  svm_iris_linear,
  data = iris,
  Petal.Width ~ Petal.Length,
  svSymbol = 17,
  dataSymbol = 20,
  symbolPalette = c(3,2,4),
  cex = 2,
  color.palette = grey.colors
)

# LINEAR - 4 FEATURES

svm_iris_linear_all <- svm(Species ~ ., data=iris, kernel="linear")
summary(svm_iris_all)

pred_all <- predict(svm_iris_all, x)
print(table(pred_all,y))

plot(
  svm_iris_linear_all,
  data=iris,
  Petal.Width ~ Petal.Length,
  svSymbol = 17,
  dataSymbol = 20,
  symbolPalette = c(3,2,4),
  cex=2,
  color.palette=grey.colors,
  slice=list(Sepal.Width=3, Sepal.Length=6.0)
)

# RADIAL - 2 FEATURES

svm_iris <- svm(Species ~ Petal.Width + Petal.Length, data=iris)
summary(svm_iris)

pred <- predict(svm_iris,x)
print(table(pred,y))

plot(
  svm_iris,
  data=iris,
  Petal.Width ~ Petal.Length,
  svSymbol=17,
  dataSymbol=20,
  symbolPalette=c(3,2,4),
  cex=2,
  color.palette=grey.colors
)

# RADIAL - 4 FEATURES

svm_iris_all <- svm(Species ~ ., data=iris)
summary(svm_iris_all)

pred_all <- predict(svm_iris_all,x)
print(table(pred_all,y))

plot(
  svm_iris_all,
  data=iris,
  Petal.Width ~ Petal.Length,
  svSymbol=17,
  dataSymbol=20,
  symbolPalette=grey.colors,
  slice=list(Sepal.Width=2.5, Sepal.Length=3.5)
)

# TUNING
set.seed(523)
svm_tune <- tune(
  svm,
  train.x=x,
  train.y=y,
  kernel="radial",
  ranges=list(cost=10^(-1:2),gamma=c(.5,1,2))
)

print(svm_tune)

svm_iris_tuned <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
summary(svm_iris_tuned)

pred_tuned <- predict(svm_iris_tuned,x)
print(table(pred_tuned,y))


