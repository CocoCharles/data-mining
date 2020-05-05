"""
Splitting data into testing and training sets and applying min max scaling
"""

import numpy as np
import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV


# Read in data
data = pd.read_csv('dcyaPreprocessed.csv', index_col=0)
print('First 5 rows of initial data:\n', data.head(), '\n')


''' Create Testing and Training sets '''

# Create dataframe for Strongly Agree
SA = data[data['IBelong'] == 1]

# Create dataframe for Agree
A = data[data['IBelong'] == 2]

# Create dataframe for Disagree
D = data[data['IBelong'] == 3]

# Create dataframe for Strongly Disagree
SD = data[data['IBelong'] == 4]


# Split training/testing sets for each answer to maintain answer distribution
ySA = SA['belong']
xSA = SA.drop(['IBelong', 'belong'], axis = 1)
xSA_train, xSA_test, ySA_train, ySA_test = train_test_split(xSA, ySA, test_size = 0.3)

yA = A['belong']
xA = A.drop(['IBelong', 'belong'], axis = 1)
xA_train, xA_test, yA_train, yA_test = train_test_split(xA, yA, test_size = 0.3)

yD = D['belong']
xD = D.drop(['IBelong', 'belong'], axis = 1)
xD_train, xD_test, yD_train, yD_test = train_test_split(xD, yD, test_size = 0.3)

ySD = SD['belong']
xSD = SD.drop(['IBelong', 'belong'], axis = 1)
xSD_train, xSD_test, ySD_train, ySD_test = train_test_split(xSD, ySD, test_size = 0.3)


# Combine to make testing/training sets

trainFrames = [xSA_train, xA_train, xD_train, xSD_train]
x_train = pd.concat(trainFrames)

ytrainFrames = [ySA_train, yA_train, yD_train, ySD_train]
y_train = pd.concat(ytrainFrames)

testFrames = [xSA_test, xA_test, xD_test, xSD_test]
x_test = pd.concat(testFrames)

ytestFrames = [ySA_test, yA_test, yD_test, ySD_test]
y_test = pd.concat(ytestFrames)


''' Apply Min Max Scaling '''

# Column Names
column_names = x_train.columns

# Min Max Scaler
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = pd.DataFrame(x_train, columns=column_names)
x_test = pd.DataFrame(x_test, columns=column_names)

# # Robust Scaler
# scaler = preprocessing.RobustScaler()
# # scaler.fit(x_train)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = pd.DataFrame(x_train, columns=column_names)
# x_test = pd.DataFrame(x_test, columns=column_names)
# print(x_train)

x_train.to_csv('xtrain.csv')
x_test.to_csv('xtest.csv')
y_train.to_csv('ytrain.csv')
y_test.to_csv('ytest.csv')