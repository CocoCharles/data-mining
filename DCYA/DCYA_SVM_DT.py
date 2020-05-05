"""
Create and test decision tree model and SVM
"""

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
from sklearn.svm import SVC


# Read in data
x_train = pd.read_csv('xtrain.csv', index_col=0)
x_test = pd.read_csv('xtest.csv', index_col=0)
y_train = pd.read_csv('ytrain.csv', index_col=0)
y_test = pd.read_csv('ytest.csv', index_col=0)
print(y_train)

# Column Names
column_names = x_train.columns


''' Decision Tree '''

# Grid Search - find best combination of parameters
# Parameter Grid
param_grid = {'criterion':['gini'], 'max_depth':[4,5,6], 'max_leaf_nodes':[5,6,7]}

# Make grid search classifier
clf_grid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, scoring='accuracy')

# Train grid search classifier
clf_grid.fit(x_test, y_test)

print("Best Parameters:\n", clf_grid.best_params_)
print('score\n', clf_grid.score(x_train, y_train))


# Create decision tree
classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, max_leaf_nodes=6)
classifier = classifier.fit(x_train, y_train)


# Display tree
# This outputs a txt file which can be copy and pasted into http://webgraphviz.com/ to display the tree
with open("belonging_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = column_names, class_names = ['agree','disagree'], filled = True, out_file=f)


# Test model
y_predictTest = classifier.predict(x_test)
y_predictTrain = classifier.predict(x_train)

print("DT Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("DT Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

# Create Confuion Matrix
print("DT Confusion Matrix:")
print(confusion_matrix(y_test, y_predictTest))

# Compute Report
print("DT Report: \n" , classification_report(y_test, y_predictTest))


''' Support Vector Machine '''

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Grid Search - find best combination of parameters
# Parameter Grid
param_grid = {'C': [0.1, 1], 'gamma': [0.01, 0.1]}

# Make grid search classifier
clf_grid = GridSearchCV(SVC(), param_grid, scoring='accuracy')

# Train grid classifier
clf_grid.fit(x_test, y_test)

print("Best Parameters:\n", clf_grid.best_params_)
print('score\n', clf_grid.score(x_train, y_train))


# SVM Model

# Create SVM
svcclassifier = SVC()
svcclassifier.fit(x_train, y_train)

# Test model
y_predictTest = svcclassifier.predict(x_test)
y_predictTrain = svcclassifier.predict(x_train)

print("SVM Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("SVM Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

# Create Confuion Matrix
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_predictTest))

# Compute Report
print("SVM Report: \n" , classification_report(y_test, y_predictTest))
