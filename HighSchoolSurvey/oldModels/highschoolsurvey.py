# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:01:29 2020

@author: LabUser
finishing svm
looking at NaN
looking into weighting
"""

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
import pydotplus

# Read in data
data = pd.read_csv('DCYA2018.csv', na_values=[' '])
print('First 5 rows of initial data:\n', data.head(), '\n')
# print('First 5 rows of initial data:\n', data.tail(), '\n')

# Drop columns
data = data.drop(data.iloc[:, 266:316], axis = 1)

# Drop open ended question (string answers)
# 'DontFitIn', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo', 'IBelong'
data = data.drop(['OtherProblems', 'Homeless2015'], axis=1)

print(data.shape)
print('First 5 rows of cleaned** data:\n', data.head(), '\n')
print('Numer of instances = %d' %data.shape[0])
print('Numer of attributes = %d' %data.shape[1])
# print('Number of Missing Values', data['IBelong'].isna().sum())
# data = data[data['IBelong'].notna()]
# print('****After****\nNumber of Missing Values', data['IBelong'].isna().sum())
# print('Numer of instances = %d' %data.shape[0])
# print('Numer of attributes = %d' %data.shape[1])

# for col in data.columns:
    #count number of missing values in each column
    # print('\t%s: %d' %(col, data[col].isna().sum()))
data = data.fillna(data.median())
# print('median: ', data.median())
# for col in data.columns:
#     #count number of missing values in each column
#     print('\t%s: %d' %(col, data[col].isna().sum()))

# def change_type (col):
for col in data.columns:
    if data.dtypes[col] == 'int64':
        data[col] = data[col].astype('int32')
    if data.dtypes[col] == 'float64':
        data[col] = data[col].astype('float32')

# print(data.dtypes)

# Function to simplify IBelong col
def simplify_belong (row):
    if row['IBelong'] == 1 or row['IBelong'] == 2:
        return '1'
    if row['IBelong'] == 3 or row['IBelong'] == 4:
        return '2'

# Add a column that converts 'IBelong' to only 1 v 2
# With 1 being agree and 2 being disagree
data['belong'] = data.apply(lambda row: simplify_belong(row), axis=1)
# print('IBelong col converted to binary:\n', data.belong, '\n', data.IBelong)

# data.to_csv('checkshit.csv')

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
xSA = SA.drop(['Schoolcode', 'IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xSA_train, xSA_test, ySA_train, ySA_test = train_test_split(xSA, ySA, test_size = 0.33)
#Check for right amount of rows- it is!
print("Total SA: ", xSA_test.shape[0] + xSA_train.shape[0], "\n  SA Test: ", xSA_test.shape[0], "\n  SA Train: ", xSA_train.shape[0])

    #Agree Represenation
#Value you are looking for
yA = A['belong']
#Dataset with missing value
xA = A.drop(['Schoolcode', 'IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xA_train, xA_test, yA_train, yA_test = train_test_split(xA, yA, test_size = 0.33)
print("\nTotal A: ", xA_test.shape[0] + xA_train.shape[0], "\n  A Test: ", xA_test.shape[0], "\n  A Train: ", xA_train.shape[0])

    #Disagree Represenation
#Value you are looking for
yD = D['belong']
#Dataset with missing value
xD = D.drop(['Schoolcode', 'IBelong', 'belong'], axis = 1)

#Split Training set (67%) and Testing set (33%)
xD_train, xD_test, yD_train, yD_test = train_test_split(xD, yD, test_size = 0.33)
print("\nTotal D: ", xD_test.shape[0] + xD_train.shape[0], "\n  D Test: ", xD_test.shape[0], "\n  D Train: ", xD_train.shape[0])

    #Strongly Disagree Represenation
#Value you are looking for
ySD = SD['belong']
#Dataset with missing value
xSD = SD.drop(['Schoolcode', 'IBelong', 'belong'], axis = 1)

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


#Create Decision Tree

classifier = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 3)
#fit the data you are using to train
classifier = classifier.fit(x_train, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['Schoolcode', 'RespondentID','IBelong','belong'], axis = 1)


#Display Tree!
with open("belonging_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = x_train.columns, class_names = ['agree','disagree'], filled = True, out_file=f)


#Predict Test Data
y_predict = classifier.predict(x_test)

#Create Confuion Matrix
print("Gini Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

#Compute Accuracy
print("Accuracy:", accuracy_score(y_test,y_predict)*100, "\n")

#Compute Report
print("Report: \n" ,
classification_report(y_test, y_predict))


#Try again with criterion as entropy

classifierE = tree.DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3)
#fit the data you are using to train
classifierE = classifierE.fit(x_train, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['Schoolcode', 'RespondentID','IBelong','belong'],axis=1)

#Display Tree!
with open("belonging_entropy.txt", "w") as f:
    f = tree.export_graphviz(classifierE, feature_names = x_train.columns, class_names = ['agree','disagree'], filled = True, out_file=f)


#Predict Test Data
y_predictE = classifierE.predict(x_test)

#Create Confuion Matrix
print("Confusion Matrix:")
confusion_matrix(y_test, y_predict)

#Compute Accuracy
print("Accuracy:", accuracy_score(y_test,y_predictE)*100, "\n") #ok so it's a tiny bit more accurate... is it worth the change?

#Compute Report
print("Report: \n" , classification_report(y_test, y_predictE))