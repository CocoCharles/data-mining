# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:01:29 2020

@author: LabUser
finishing svm
looking into weighting
http://webgraphviz.com/

looking at NaN - k nearest neighbor
removing records - must be scientific
scaling - could try
look into outliers?
split up the classifaction rather than range
we need more consulting with expert / social worker
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
data = pd.read_csv('DCYApreprocessed.csv', index_col=0)
print('First 5 rows of initial data:\n', data.head(), '\n')
print('Numer of instances = %d' %data.shape[0])
print('Numer of attributes = %d' %data.shape[1])


#Create table for Strongly Agree
SA = data[data['IBelong'] == 1]

#Create table for Agree
A = data[data['IBelong'] == 2]

#Create table for Disagree
D = data[data['IBelong'] == 3]

#Create table for Strongly Disagree
SD = data[data['IBelong'] == 4]

    #Strongly Agree Represenation
#Value you are looking for
ySA = SA['belong']
#Dataset with missing value
xSA = SA.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xSA_train, xSA_test, ySA_train, ySA_test = train_test_split(xSA, ySA, test_size = 0.3)
#Check for right amount of rows- it is!
print("Total SA: ", xSA_test.shape[0] + xSA_train.shape[0], "\n  SA Test: ", xSA_test.shape[0], "\n  SA Train: ", xSA_train.shape[0])

    #Agree Represenation
#Value you are looking for
yA = A['belong']
#Dataset with missing value
xA = A.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xA_train, xA_test, yA_train, yA_test = train_test_split(xA, yA, test_size = 0.3)
print("\nTotal A: ", xA_test.shape[0] + xA_train.shape[0], "\n  A Test: ", xA_test.shape[0], "\n  A Train: ", xA_train.shape[0])

    #Disagree Represenation
#Value you are looking for
yD = D['belong']
#Dataset with missing value
xD = D.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xD_train, xD_test, yD_train, yD_test = train_test_split(xD, yD, test_size = 0.3)
print("\nTotal D: ", xD_test.shape[0] + xD_train.shape[0], "\n  D Test: ", xD_test.shape[0], "\n  D Train: ", xD_train.shape[0])

    #Strongly Disagree Represenation
#Value you are looking for
ySD = SD['belong']
#Dataset with missing value
xSD = SD.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xSD_train, xSD_test, ySD_train, ySD_test = train_test_split(xSD, ySD, test_size = 0.3)
print("\nTotal SD: ", xSD_test.shape[0] + xSD_train.shape[0], "\n  SD Test: ", xSD_test.shape[0], "\n  SD Train: ", xSD_train.shape[0])


#Create training data: combine samples for your X and Y

#Create variable holding all training data sets with data being classified
trainFrames = [xSA_train, xA_train, xD_train, xSD_train]
#Combine frames to make training sample
x_train = pd.concat(trainFrames)

#Create variable holding all training data sets for predicted data
ytrainFrames = [ySA_train, yA_train, yD_train, ySD_train]
#Combine frames to make training sample
y_train = pd.concat(ytrainFrames)


#Create Testing Data: combine samples for your X and Y 

#Create variable holding all testing data sets with data being classified
testFrames = [xSA_test, xA_test, xD_test, xSD_test]
#Combine frames to make testing sample
x_test = pd.concat(testFrames)

#Create variable holding all testing data sets with data being classified
ytestFrames = [ySA_test, yA_test, yD_test, ySD_test]
#Combine frames to make testing sample
y_test = pd.concat(ytestFrames)


#Check if data is split correctly

print("Number of Rows in Training Sample (X): ", x_train.shape[0])
print("Number of Rows in Training Sample (Y): ", y_train.shape[0])

print("\nNumber of Rows in Testing Sample (X): ", x_test.shape[0])
print("Number of Rows in Testing Sample (Y): ", y_test.shape[0])

#test if you have the right number of rows:
print("\nTotal Data: ", y_test.shape[0]+y_train.shape[0])

#Result: x_train, y_train, x_test, and y_test data sets

print('\nxtrain prior to Max Min scaling\n', x_train)

# Column Names
column_names = x_train.columns

# Min Max Scaler
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = pd.DataFrame(x_train, columns=column_names)
print('\nxtrain post Max Min scaling\n', x_train)

# # Robust Scaler
# scaler = preprocessing.RobustScaler()
# # scaler.fit(x_train)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train)



#Create Decision Tree

classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth = 5, random_state = 100, max_leaf_nodes=19)
#fit the data you are using to train
classifier = classifier.fit(x_train, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['IBelong','belong'], axis = 1)


#Display Tree!
with open("belonging_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = column_names, class_names = ['agree','disagree'], filled = True, out_file=f)


#Predict Test Data
y_predictTest = classifier.predict(x_test)
y_predictTrain = classifier.predict(x_train)

print("DT Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("DT Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

#Create Confuion Matrix
print("DT Confusion Matrix:")
print(confusion_matrix(y_test, y_predictTest))

#Compute Report
print("DT Report: \n" , classification_report(y_test, y_predictTest))


# # Grid Search
# # Parameter Grid
# # param_grid = {'C': [0.1, 1], 'gamma': [1, 0.1]}   // 1 and 0.1 ***good results 98.2 87.3
# # param_grid = {'C': [10, 1], 'gamma': [0.01, 0.1]} // 10 and 0.01 89.2 87.6
# param_grid = {'C': [10, 1, 0.1], 'gamma': [0.01, 0.1, 1]} 1 .01 93.8 87.4
# c 10 and g scale 99.1 and 85.9

# # Make grid search classifier
# clf_grid = GridSearchCV(SVC(), param_grid, scoring='accuracy')

# # Train the classifier
# clf_grid.fit(x_test, y_test)

# # clf = grid.best_estimator_()
# print("Best Parameters:\n", clf_grid.best_params_)
# print("Best Estimators:\n", clf_grid.best_estimator_)


# # SVM Model

# # Create SVM (using the support vector classifier class - SVC)
# svcclassifier = SVC(kernel='poly')
# svcclassifier.fit(x_train, y_train)

# #Predict Test Data
# y_predictTest = svcclassifier.predict(x_test)
# y_predictTrain = svcclassifier.predict(x_train)

# print("SVM Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
# print("SVM Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

# #Create Confuion Matrix
# print("SVM Confusion Matrix:")
# print(confusion_matrix(y_test, y_predictTest))

# #Compute Report
# print("SVM Report: \n" , classification_report(y_test, y_predictTest))
