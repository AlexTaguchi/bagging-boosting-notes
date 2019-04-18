# bagging-boosting-notes
Collection of notes on bagging and boosting machine learning algorithms

## Outline
Bagging and boosting are ensemble methods for improving the generality of machine learning algorithms. The classic example is a decision tree, which is a weak learner when implemented by itself. Various approaches for bagging and boosting decision trees are given below.

## Bagging
### Random Forest
Train decision trees are grown on random subsets of the data, and results are averaged.

## Boosting
### Adaboost (Adaptive Boosting)
The weak learners in AdaBoost are usually decision trees with a single split, called decision stumps. An initial decision stump is generated with each sample weighted equally in the training process. The weights are then adjusted such that incorrectly classified samples are increased (and correctly classified ones decreased), and a new decision stump is generated with a stronger bias towards correctly classifying the previously incorrectly classified samples. Each decision stump is additionally scored according to its ability to correct the previous decision stumps mistakes.

Reference: https://www.youtube.com/watch?v=LsK-xG1cLYA

### Gradient Boosting
The weak learners are typically not decision stumps in this case, nor are they fully grown decision trees. In the simplest case, Gradient Boosting is a machine learning regressor based on iteratively correcting the residuals of the previous tree. At first the average of all labels is guessed as the initial solution. The residual differences with the ground truth labels are calculated, and a new regression tree is generated, where leaves corresponding to multiple labels take on the average value. The result of the trees is multiplied with a learning rate (0-1), and more trees are grown to iteratively improve the residuals.

Reference: https://www.youtube.com/watch?v=3CC4N4z3GJc

### XGBoost
A scalable version of gradient boosting that speeds up the learning process ten-fold with a fast approximation for finding the best tree split, among other parallelized implementations.

#### Install XGBoost
```bash
pip install xgboost
```
or
```bash
conda install -c conda-forge xgboost
```

#### Download diabetes dataset
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

#### Training example
```python
# Import modules
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

# Split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit model
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

Reference: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
