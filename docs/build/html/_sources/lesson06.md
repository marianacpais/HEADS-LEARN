# Lesson 06 - Ensemble Models - Bagging, Boosting and Reliability

## Summary

"Ensemble methods aim at improving the predictive performance of a given statistical learning or model fitting technique." Buhlmann (2012)

"The general principle of ensemble methods is to construct a linear combination of some model fitting method, instead of using a single fit of the method." Buhlmann (2012)

### Bagging

Bootstrap aggregating is an ensemble method for improving unstable estimation or classification schemes.
Introduced as a variance reduction mechanism for methods that do variable selection and fitting in a linear model.
Popular due to it simplicity and popularity of bootstrap methods.
Breiman (1996)

**Algorithm**:
1. Create a bootstrapped sample of same size as original sample.
2. Compute the bootstrapped estimator (e.g. model derivation and error)
3. Repeat steps 1 and 2, M times, usually M = 50 or 100.
The bagged estimator is the average of all estimators (if M = inf then we would have an expected value for the estimator).
The finite number M in practice governs the accuracy of the Monte Carlo approximation but otherwise, it shouldn’t be viewed as a tuning parameter for bagging.

For classification, two options apply:
- Compute the average of computed probabilities of each model
- Majority voting (proposed by Breiman)
Bagging improves predictive performance of classification and regression trees.
It is worth pointing out that bagging a decision tree is almost never worse (in terms of predictive power) than a single tree.

**Variants:**
- Subagging: Subsample aggregating
- Bragging: Bootstrap robust aggregating

**Disadvantages:**
- The main disadvantage of bagging, and other ensemble algorithms, is the lack of interpretation.
- A linear combination of decision trees is much harder to interpret than a single tree.
- Likewise bagging a variable selection-fitting algorithm for linear models gives little clues which of the predictor variables are actually important.

### Boosting

Unlike bagging, which is a parallel ensemble method, boosting methods are sequential ensemble algorithms where the weights given to learned examples depend on the previous fitted functions.
Boosting has been empirically demonstrated to be very accurate in terms of classification (e.
g. AdaBoost algorithm)

Schapire (1990) and Freund (1995)

Boosting algorithms have often better predictive power than bagging.
For all data-sets tested by Breiman, boosting trees was better than a single classification tree.
The biggest loss for boosting in comparison with bagging was for a data-set with very low misclassification error.

Breiman (1998)

**Algorithm:**
1. Learn the base model with the data
2. Compute the gradient vector (loss function attributed to each case)
3. Repeat 1 and 2 with weighted data according to its error contribution

The key idea is to give more weight to instances for which the error is higher, to improve the model in that direction.

### Stacking

Stacking works by combining heterogeneous models learned on the same original data set.

After learning the desired models, the outputs of the single models are combined using
another model which takes them as inputs.

For example:
- Learn a decision tree, a k-nn and an SVM with the data
- Stack the outputs of the three models using a linear model
