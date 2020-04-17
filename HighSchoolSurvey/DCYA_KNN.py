# -*- coding: utf-8 -*-
"""
Preprocessing data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
from sklearn.svm import SVC
import pydotplus


# Read in data
data_original = pd.read_csv('actualytest.csv', na_values=[' '], index_col=0)
# print('First 5 rows of initial data:\n', data.head(), '\n')



# #Split Training set (67%) and Testing set (33%)
# train, test = train_test_split(data_original, test_size = 0.33)
# #Check for right amount of rows
# print("Total: ", train.shape[0] + test.shape[0], "\nTest: ", test.shape[0], "\nTrain: ", train.shape[0])

# # datasets = [test, train]


# # Imputing missing values using K Nearest Neighbor Imputer
# def apply_knn (data):

#     # Column names
#     data_columns = data.columns

#     loops = int(data.shape[0] / 1000)
#     testsize = int(data.shape[0] / loops)
#     nNeighbors = int(np.sqrt(data.shape[0]) / 2)

#     # while data.shape[0] > 1000:
#     print('data shape: ', data.shape, '\nloops: ', loops, '\ntestsize: ', testsize, '\n neighbors: ', nNeighbors)

#     # Split first 1/16 (1018 records)
#     remaining, current = train_test_split(data, test_size = testsize)
#     print('Remaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)


#     for x in range(loops):
#         if x == 0:
#             # Imputation of missing values (returns array)
#             imputer = KNNImputer(n_neighbors=nNeighbors)
#             fill_data = imputer.fit_transform(current)
#             print(fill_data)

#             # Round to nearest integer
#             fill_data = np.round(fill_data)
#             print(fill_data)

#             # Convert to dataframe
#             data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')
#             print(data)
#         elif x == (loops - 1):
#             current = remaining
#             print('\nRemaining shape: 0', '\nCurrent shape: ', current.shape)

#             # Imputation of missing values (returns array)
#             imputer = KNNImputer(n_neighbors=nNeighbors)
#             fill_data = imputer.fit_transform(current)

#             # Round to nearest integer
#             fill_data = np.round(fill_data)

#             # Convert to dataframe
#             add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

#             # Concatenate frames
#             data = pd.concat([data, add_data], ignore_index=True)
#         else:
#             remaining, current = train_test_split(remaining, test_size = testsize)
#             print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

#             # Imputation of missing values (returns array)
#             imputer = KNNImputer(n_neighbors=nNeighbors)
#             fill_data = imputer.fit_transform(current)

#             # Round to nearest integer
#             fill_data = np.round(fill_data)

#             # Convert to dataframe
#             add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

#             # Concatenate frames
#             data = pd.concat([data, add_data], ignore_index=True)
#     return data


# x = apply_knn(train)
# y = apply_knn(test)

# x.to_csv('x.csv')
# y.to_csv('y.csv')

# print(x, y)


# data = apply_knn(data_original)
# data.to_csv('y.csv')
data = pd.read_csv('y.csv', index_col=0)
print(data)

# x = pd.read_csv('x.csv', index_col=0)
# y = pd.read_csv('y.csv', index_col=0)
# print(x)
# print(y)


#Split Training set (67%) and Testing set (33%)
train, test = train_test_split(data, test_size = 0.33)
#Check for right amount of rows
print("Total: ", train.shape[0] + test.shape[0], "\nTest: ", test.shape[0], "\nTrain: ", train.shape[0])


    #Strongly Agree Represenation
#Value you are looking for
x_train = train['belong']
#Dataset with missing value
y_train = train.drop(['IBelong', 'belong'], axis = 1)

#Value you are looking for
x_test = test['belong']
#Dataset with missing value
t_test = test.drop(['IBelong', 'belong'], axis = 1)


# Column Names
column_names = x_train.columns


#Create Decision Tree

classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth = 5, random_state = 100, max_leaf_nodes=19)
#fit the data you are using to train
model = classifier.fit(x_train, x_test)

#Display Tree!
with open("belonging_gini.txt", "w") as f:
    f = tree.export_graphviz(model, feature_names = column_names, class_names = ['agree','disagree'], filled = True, out_file=f)


#Predict Test Data
y_predictTest = model.predict(y_train)
# y_predictTrain = classifier.predict(x_train)

print(y_predictTest)
# print("DT Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("DT Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

#Create Confuion Matrix
print("DT Confusion Matrix:")
print(confusion_matrix(y_test, y_predictTest))

#Compute Report
print("DT Report: \n" , classification_report(y_test, y_predictTest))

# #Create Decision Tree

# classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth = 5, random_state = 100, max_leaf_nodes=19)
# #fit the data you are using to train
# classifier = classifier.fit(x_train, y_train)


# #Display Tree!
# with open("belonging_gini.txt", "w") as f:
#     f = tree.export_graphviz(classifier, feature_names = column_names, class_names = ['agree','disagree'], filled = True, out_file=f)


# #Predict Test Data
# y_predictTest = classifier.predict(x_test)
# y_predictTrain = classifier.predict(x_train)

# print("DT Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
# print("DT Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

# #Create Confuion Matrix
# print("DT Confusion Matrix:")
# print(confusion_matrix(y_test, y_predictTest))

# #Compute Report
# print("DT Report: \n" , classification_report(y_test, y_predictTest))
