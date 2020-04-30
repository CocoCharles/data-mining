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
data = pd.read_csv('DCYApreprocessed63.csv', index_col=0)
print('First 5 rows of initial data:\n', data.head(), '\n')
print(data.shape)
# print('Numer of instances = %d' %data.shape[0])
# print('Numer of attributes = %d' %data.shape[1])

# # Drop rows and columns
# data = data.drop(data.index[1000:16288])
# data = data_original.drop(data_original.iloc[:, 266:316], axis = 1)

# # Drop open ended question (string answers)
# # 'DontFitIn', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo', 'IBelong', 'DateSexRecode'
# data = data.drop(['Year', 'Schoolcode', 'RespondentID', 'Weighting2', 'OtherProblems', 'Homeless2015', 'HeightInch', 'HeightFeet', 'Weight'], axis=1)


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
xSA_train, xSA_test, ySA_train, ySA_test = train_test_split(xSA, ySA, test_size = 0.33)
#Check for right amount of rows- it is!
print("Total SA: ", xSA_test.shape[0] + xSA_train.shape[0], "\n  SA Test: ", xSA_test.shape[0], "\n  SA Train: ", xSA_train.shape[0])

    #Agree Represenation
#Value you are looking for
yA = A['belong']
#Dataset with missing value
xA = A.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xA_train, xA_test, yA_train, yA_test = train_test_split(xA, yA, test_size = 0.33)
print("\nTotal A: ", xA_test.shape[0] + xA_train.shape[0], "\n  A Test: ", xA_test.shape[0], "\n  A Train: ", xA_train.shape[0])

    #Disagree Represenation
#Value you are looking for
yD = D['belong']
#Dataset with missing value
xD = D.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xD_train, xD_test, yD_train, yD_test = train_test_split(xD, yD, test_size = 0.33)
print("\nTotal D: ", xD_test.shape[0] + xD_train.shape[0], "\n  D Test: ", xD_test.shape[0], "\n  D Train: ", xD_train.shape[0])

    #Strongly Disagree Represenation
#Value you are looking for
ySD = SD['belong']
#Dataset with missing value
xSD = SD.drop(['IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xSD_train, xSD_test, ySD_train, ySD_test = train_test_split(xSD, ySD, test_size = 0.33)
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
print("Number of Rows in Training Sample (Y): ", y_train.shape[0]) #they match!

print("\nNumber of Rows in Testing Sample (X): ", x_test.shape[0])
print("Number of Rows in Testing Sample (Y): ", y_test.shape[0]) #these match too!

#test if you have the right number of rows:
print("\nTotal Data: ", y_test.shape[0]+y_train.shape[0])

#Result: x_train, y_train, x_test, and y_test data sets


# Min Max Scaler
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)

x_train1 = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train1)

# # Robust Scaler
# scaler = preprocessing.RobustScaler()
# # scaler.fit(x_train)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train)


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
classifier = classifier.fit(x_train1, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['IBelong','belong'], axis = 1)


#Display Tree!
with open("belonging_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = x_train.columns, class_names = ['agree','disagree'], filled = True, out_file=f)


#Predict Test Data
y_predictTest = classifier.predict(x_test)
y_predictTrain = classifier.predict(x_train1)

print("Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

#Create Confuion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predictTest))

#Compute Report
print("Report: \n" , classification_report(y_test, y_predictTest))

