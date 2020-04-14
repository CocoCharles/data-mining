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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
from sklearn.svm import SVC
import pydotplus

# Read in data
data_original = pd.read_csv('DCYA2018.csv', na_values=[' '])
# print('First 5 rows of initial data:\n', data.head(), '\n')

# Drop columns
data = data_original.drop(data_original.iloc[:, 266:316], axis = 1)
data = data.drop(data.index[1000:16288])

# Drop open ended question (string answers)
# 'DontFitIn', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo', 'IBelong', 'DateSexRecode'
data = data.drop(['Year', 'Schoolcode', 'RespondentID', 'Weighting2', 'OtherProblems', 'Homeless2015', 'HeightInch', 'HeightFeet', 'Weight'], axis=1)

print(data.shape)
# print('First 5 rows of cleaned** data:\n', data.head(), '\n')
# print('Numer of instances = %d' %data.shape[0])
# print('Numer of attributes = %d' %data.shape[1])


# # Create column for sum of unanswered questions
# data['NotAnswered'] = data.isna().sum(axis=1)

# # Keep rows with at least 25% (66) of questions - consider as noise ***
# data = data[data.NotAnswered < 200]
# # print('\t >200 = %.2f' % (data['NotAnswered'] > 200).sum())
# data = data.drop(['NotAnswered'], axis = 1)

data.loc[(data['Abstinence'] == 15), 'NoIntercourse'] = 15
# print(data.NoIntercourse)

# Recode columns with binary values to 0(no) and 1(yes)
for col in data.columns:
    if 0 in data[col].unique():
        data.loc[(data[col] > 0), col] = 1
        # data[col] = data[col].fillna(0.0)


# for col in data.columns:
#     if data[col].isna().sum() > 3000:
#         data[col] = data[col].fillna(0.0)

# data = data.dropna()
# print(data.shape)

# for col in data.columns:
#     # count number of missing values in each column
#     print('\t%s: %d' %(col, data[col].isna().sum()))


# nan_data = data[data['IBelong'].isna()]
# print("nan_data:\n", nan_data.shape)

# belong_data = data.dropna(subset=['IBelong'])
# print(belong_data.shape)


# #Dataset with missing value
# x_train = belong_data.drop(['IBelong'], axis = 1)
# #Value you are looking for
# y_train = belong_data['IBelong']
# # x test
# x_test = nan_data.drop(['IBelong'], axis = 1)



# from sklearn.neighbors import KNeighborsClassifier
# KNNclassifier = KNeighborsClassifier(n_neighbors = 5)
# KNNclassifier.fit(x_train, y_train)

# y_pred = KNNclassifier.predict(x_test)
# print(y_pred)

# print(data)

# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# fill_data = imputer.fit_transform(data)
# fill_data = np.round(fill_data)
# print(fill_data)


# K NEAREST NEIGHBOR ***
# Replace NaN with mode
for col in data.columns:
    data[col] = data[col].fillna(data[col].mode()[0])
# data = data.fillna(value = data.mode())
# print('First 5 rows of cleaned** data:\n', data.head(), '\n')


# def change_type (col):
for col in data.columns:
    if data.dtypes[col] == 'int64':
        data[col] = data[col].astype('int32')
    if data.dtypes[col] == 'float64':
        data[col] = data[col].astype('float32')

# print(data.dtypes)


#Find Number of each value
# print("Number of each value\n",data['IBelong'].value_counts())

#Create 2 classes: those who feel like they belong and those who do not
belong_map = {1.0:1,
              2.0:1,
              3.0:2,
              4.0:2}
                 
data["belong"] = data["IBelong"].map(belong_map)
# data['belong'] = data.belong.astype('float32')

print("\nNumber of each value (belong):\n")
print(data['belong'].value_counts())

sex_protection_map = {1.0:1, 2.0:1, 3.0:1, 4.0:1, 5.0:1, 6.0:1, 7.0:1, 8.0:1, 9.0:1, 10.0:1, 11.0:1, 12.0:1, 15.0:0}
for col in data.columns:
    if 15 in data[col].unique():
        data[col] = data[col].map(sex_protection_map)
# data["Abstinence"] = data["Abstinence"].map(sex_protection_map)
# data['Abstinence'] = data.Abstinence.astype('float32')

print(data['NoIntercourse'].value_counts())
print(data)
data.to_csv('sampleDCYA.csv')


# Add a column that converts 'Race' to only 1 v 2
# With 1 being POC and 2 being white
# race_map = {1.0:1.0, 2.0:1.0, 3.0:1.0, 4.0:1.0, 5.0:1.0, 6.0:1.0, 7.0:1.0, 8.0:2.0, 9.0:1.0}
# data["Race"] = data["Race"].map(race_map)
# data['Race'] = data.Race.astype('float32')
# print('Value counts for race:\n', data.Race.value_counts())


# for col in data.columns:
#     if data[col].nunique() == 1:
#         print(col)
#     # print(col, ': ', data[col].nunique())


# #Create table for Strongly Agree
# SA = data[data['IBelong'] == 1]

# #Create table for Agree
# A = data[data['IBelong'] == 2]

# #Create table for Disagree
# D = data[data['IBelong'] == 3]

# #Create table for Strongly Disagree
# SD = data[data['IBelong'] == 4]

#     #Strongly Agree Represenation
# #Value you are looking for
# ySA = SA['belong']
# #Dataset with missing value
# xSA = SA.drop(['IBelong', 'belong'], axis = 1)

# #Split Training set (67%) and Testing set (33%)
# xSA_train, xSA_test, ySA_train, ySA_test = train_test_split(xSA, ySA, test_size = 0.33)
# #Check for right amount of rows- it is!
# print("Total SA: ", xSA_test.shape[0] + xSA_train.shape[0], "\n  SA Test: ", xSA_test.shape[0], "\n  SA Train: ", xSA_train.shape[0])

#     #Agree Represenation
# #Value you are looking for
# yA = A['belong']
# #Dataset with missing value
# xA = A.drop(['IBelong', 'belong'], axis = 1)

# #Split Training set (67%) and Testing set (33%)
# xA_train, xA_test, yA_train, yA_test = train_test_split(xA, yA, test_size = 0.33)
# print("\nTotal A: ", xA_test.shape[0] + xA_train.shape[0], "\n  A Test: ", xA_test.shape[0], "\n  A Train: ", xA_train.shape[0])

#     #Disagree Represenation
# #Value you are looking for
# yD = D['belong']
# #Dataset with missing value
# xD = D.drop(['IBelong', 'belong'], axis = 1)

# #Split Training set (67%) and Testing set (33%)
# xD_train, xD_test, yD_train, yD_test = train_test_split(xD, yD, test_size = 0.33)
# print("\nTotal D: ", xD_test.shape[0] + xD_train.shape[0], "\n  D Test: ", xD_test.shape[0], "\n  D Train: ", xD_train.shape[0])

#     #Strongly Disagree Represenation
# #Value you are looking for
# ySD = SD['belong']
# #Dataset with missing value
# xSD = SD.drop(['IBelong', 'belong'], axis = 1)

# #Split Training set (67%) and Testing set (33%)
# xSD_train, xSD_test, ySD_train, ySD_test = train_test_split(xSD, ySD, test_size = 0.33)
# print("\nTotal SD: ", xSD_test.shape[0] + xSD_train.shape[0], "\n  SD Test: ", xSD_test.shape[0], "\n  SD Train: ", xSD_train.shape[0])


# #Create training data: combine samples for your X and Y

# #Create variable holding all training data sets with data being classified
# trainFrames = [xSA_train, xA_train, xD_train, xSD_train]
# #Combine frames to make training sample
# x_train = pd.concat(trainFrames)

# #Create variable holding all training data sets for predicted data
# ytrainFrames = [ySA_train, yA_train, yD_train, ySD_train]
# #Combine frames to make training sample
# y_train = pd.concat(ytrainFrames)


# #Create Testing Data: combine samples for your X and Y 

# #Create variable holding all testing data sets with data being classified
# testFrames = [xSA_test, xA_test, xD_test, xSD_test]
# #Combine frames to make testing sample
# x_test = pd.concat(testFrames)

# #Create variable holding all testing data sets with data being classified
# ytestFrames = [ySA_test, yA_test, yD_test, ySD_test]
# #Combine frames to make testing sample
# y_test = pd.concat(ytestFrames)


# #Check if data is split correctly

# print("Number of Rows in Training Sample (X): ", x_train.shape[0])
# print("Number of Rows in Training Sample (Y): ", y_train.shape[0])

# print("\nNumber of Rows in Testing Sample (X): ", x_test.shape[0])
# print("Number of Rows in Testing Sample (Y): ", y_test.shape[0])

# #test if you have the right number of rows:
# print("\nTotal Data: ", y_test.shape[0]+y_train.shape[0])

# #Result: x_train, y_train, x_test, and y_test data sets


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,3))
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# X_test = scaler.transform(x_test)

# print(x_train)



#Principal Component Analysis
#Components created: 1. Linear Combinations of original attributes
#                    2. Perpendicular to each other
#                    3. Capture the maximum amount of variation in the data.
from sklearn.decomposition import PCA

# # intialize pca and logistic regression model
# pca = PCA(n_components=100)

# # fit and transform data
# x_train = pca.fit_transform(x_train)
# print(x_train)
# x_test = pca.transform(x_test)
# print(x_test)



# # SVM Model

# # Create SVM (using the support vector classifier class - SVC)
# svcclassifier = SVC(kernel='rbf')
# svcclassifier.fit(x_train, y_train)

# #Predict Test Data
# y_predictTest = svcclassifier.predict(x_test)
# y_predictTrain = svcclassifier.predict(x_train)

# print("Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
# print("Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

# #Create Confuion Matrix
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_predictTest))

# #Compute Report
# print("Report: \n" , classification_report(y_test, y_predictTest))

