"""
Preprocessing data
"""

import numpy as np
import pandas as pd

import sklearn
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV


# Read in data
data_original = pd.read_csv('DCYA2018.csv', na_values=[' '])
print('First 5 rows of initial data:\n', data_original.head(), '\n')

# Drop columns that are not questions
data = data_original.drop(data_original.iloc[:, 266:316], axis = 1)

# Drop questions that are irrelevant
data = data.drop(['Year', 'Schoolcode', 'RespondentID', 'Weighting2', 'OtherProblems', 'Homeless2015', 'HeightInch', 'HeightFeet', 'Weight'], axis=1)
print(data.shape)

# Add values missing from NoIntercourse
data.loc[(data['Abstinence'] == 15), 'NoIntercourse'] = 15


# Recode columns with binary values to 0(no) and 1(yes)
for col in data.columns:
    if 0 in data[col].unique():
        data.loc[(data[col] > 0), col] = 1


# Change types from 64 to 32
for col in data.columns:
    if data.dtypes[col] == 'int64':
        data[col] = data[col].astype('int32')
    if data.dtypes[col] == 'float64':
        data[col] = data[col].astype('float32')


# Add a column that converts 'IBelong' to binary
# With 1 being agree and 2 being disagree
belong_map = {1.0:1,
              2.0:1,
              3.0:2,
              4.0:2}
                 
data['belong'] = data['IBelong'].map(belong_map)


# Recode attributes of the 'protected sex' question to binary 0 and 1
sex_protection_map = {1.0:1, 2.0:1, 3.0:1, 4.0:1, 5.0:1, 6.0:1, 7.0:1, 8.0:1, 9.0:1, 10.0:1, 11.0:1, 12.0:1, 15.0:0}
for col in data.columns:
    if 15 in data[col].unique():
        data[col] = data[col].map(sex_protection_map)


# Replaces missing values in sections where students were told to skip if not applicable to them with 0
for col in data.columns:
    if data[col].isna().sum() > 6000:
        data[col] = data[col].fillna(0.0)

print('\nCleaned data with NaNs:\n', data)


# Imputing missing values using K Nearest Neighbor Imputer
def apply_knn (data):

    # Column names
    data_columns = data.columns

    loops = int(data.shape[0] / 1000)
    testsize = int(data.shape[0] / loops)
    nNeighbors = int(np.sqrt(data.shape[0]) / 2)

    # while data.shape[0] > 1000:
    print('data shape: ', data.shape, '\nloops: ', loops, '\ntestsize: ', testsize, '\n neighbors: ', nNeighbors)

    # Split first 1/16 (1018 records)
    remaining, current = train_test_split(data, test_size = testsize)
    print('Remaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)


    for x in range(loops):
        if x == 0:
            # Imputation of missing values (returns array)
            imputer = KNNImputer(n_neighbors=nNeighbors)
            fill_data = imputer.fit_transform(current)

            # Round to nearest integer
            fill_data = np.round(fill_data)

            # Convert to dataframe
            data = pd.DataFrame(fill_data, columns=data_columns, dtype='float32')
        elif x == (loops - 1):
            current = remaining
            print('\nRemaining shape: 0', '\nCurrent shape: ', current.shape)

            # Imputation of missing values (returns array)
            imputer = KNNImputer(n_neighbors=nNeighbors)
            fill_data = imputer.fit_transform(current)

            # Round to nearest integer
            fill_data = np.round(fill_data)

            # Convert to dataframe
            add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='float32')

            # Concatenate frames
            data = pd.concat([data, add_data], ignore_index=True)
        else:
            remaining, current = train_test_split(remaining, test_size = testsize)
            print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

            # Imputation of missing values (returns array)
            imputer = KNNImputer(n_neighbors=nNeighbors)
            fill_data = imputer.fit_transform(current)

            # Round to nearest integer
            fill_data = np.round(fill_data)

            # Convert to dataframe
            add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='float32')

            # Concatenate frames
            data = pd.concat([data, add_data], ignore_index=True)
    return data


# Apply KNN
data = apply_knn(data)
print('Preprocessed data:\n', data)

data.to_csv('dcyaPreprocessed.csv')