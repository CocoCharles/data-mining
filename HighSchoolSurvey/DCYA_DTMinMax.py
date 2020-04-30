# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:01:29 2020

@author: LabUser
finishing svm
looking at NaN
looking into weighting
http://webgraphviz.com/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
from sklearn.svm import SVC
import pydotplus

# Read in data
x_train = pd.read_csv('x_train.csv', index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0)
x_test = pd.read_csv('x_test.csv', index_col=0)
y_test = pd.read_csv('y_test.csv', index_col=0)

print('First 5 rows of initial x_train:\n', x_train.head(), '\n')
print('First 5 rows of initial y_train:\n', y_train.head(), '\n')
print('First 5 rows of initial x_test:\n', x_test.head(), '\n')
print('First 5 rows of initial y_test:\n', y_test.head(), '\n')
# print('Numer of instances = %d' %data.shape[0])
# print('Numer of attributes = %d' %data.shape[1])

# # Drop rows and columns
# data = data.drop(data.index[1000:16288])
# data = data_original.drop(data_original.iloc[:, 266:316], axis = 1)

# # Drop open ended question (string answers)
# # 'DontFitIn', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo', 'IBelong', 'DateSexRecode'
# data = data.drop(['Year', 'Schoolcode', 'RespondentID', 'Weighting2', 'OtherProblems', 'Homeless2015', 'HeightInch', 'HeightFeet', 'Weight'], axis=1)



#Check if data is split correctly

print("Number of Rows in Training Sample (X): ", x_train.shape[0])
print("Number of Rows in Training Sample (Y): ", y_train.shape[0]) #they match!

print("\nNumber of Rows in Testing Sample (X): ", x_test.shape[0])
print("Number of Rows in Testing Sample (Y): ", y_test.shape[0]) #these match too!

#test if you have the right number of rows:
print("\nTotal Data: ", y_test.shape[0]+y_train.shape[0])

#Result: x_train, y_train, x_test, and y_test data sets


# # Principal Component Analysis
# # Components created: 1. Linear Combinations of original attributes
# #                    2. Perpendicular to each other
# #                    3. Capture the maximum amount of variation in the data.
# from sklearn.decomposition import PCA

# # intialize pca and logistic regression model
# pca = PCA(n_components=60)

# # fit and transform data
# x_train = pca.fit_transform(x_train)
# print(x_train)
# x_test = pca.transform(x_test)
# print(x_test)


#Create Decision Tree

classifier = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 5)
#fit the data you are using to train
classifier = classifier.fit(x_train, y_train)


#Display Tree!
with open("belonging_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = x_train.columns, class_names = ['agree','disagree'], filled = True, out_file=f)


#Predict Test Data
y_predictTest = classifier.predict(x_test)
y_predictTrain = classifier.predict(x_train)

print("Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

#Create Confuion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predictTest))

#Compute Report
print("Report: \n" , classification_report(y_test, y_predictTest))

