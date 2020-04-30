# -*- coding: utf-8 -*-
"""
Preprocessing data
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import sklearn
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
# from sklearn import tree
# from sklearn.svm import SVC
# import pydotplus


# Read in data
# data_original = pd.read_csv('DCYA2018.csv', na_values=[' '])

y_test = pd.read_csv('y_test.csv', index_col=0)

print('First 5 rows of initial y_test:\n', y_test.head(), '\n')


# print('\nCleaned data with NaNs:\n', data)
# data.to_csv('DCYAmode.csv')


# Imputing missing values using K Nearest Neighbor Imputer

# Column names
y_test_columns = y_test.columns


# Split first 1/9 (921 records)
remaining, current = train_test_split(y_test, test_size = 921)
print('Remaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=63)
fill_data = imputer.fit_transform(current)
print(fill_data)

# Convert to dataframe
data = pd.DataFrame(fill_data, columns=y_test_columns)
print(data)


# Split 2
remaining, current = train_test_split(remaining, test_size = 921)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=63)
fill_data = imputer.fit_transform(current)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=y_test_columns)

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 3
remaining, current = train_test_split(remaining, test_size = 921)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=63)
fill_data = imputer.fit_transform(current)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=y_test_columns)

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 4
remaining, current = train_test_split(remaining, test_size = 921)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=63)
fill_data = imputer.fit_transform(current)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=y_test_columns)

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 5
current = remaining
print('\nRemaining shape: 0', '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=63)
fill_data = imputer.fit_transform(current)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=y_test_columns)

# Concatenate frames
y_test = pd.concat([data, add_data], ignore_index=True)

print(y_test)
y_test.to_csv('y_test.csv')
# print(y_test.belong.unique())