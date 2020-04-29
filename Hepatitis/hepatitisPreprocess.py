'''
Project 2
Summarize each attribute
Load and preprocess data

Hepatitis Dataset
155 records | 19 attributes + 1 for variable live or die

Attribute Information:

1. Class: DIE, LIVE
2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
3. SEX: male, female
4. STEROID: no, yes
5. ANTIVIRALS: no, yes
6. FATIGUE: no, yes
7. MALAISE: no, yes
8. ANOREXIA: no, yes
9. LIVER BIG: no, yes
10. LIVER FIRM: no, yes
11. SPLEEN PALPABLE: no, yes
12. SPIDERS: no, yes
13. ASCITES: no, yes
14. VARICES: no, yes
15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
-- see the note below
16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
17. SGOT: 13, 100, 200, 300, 400, 500,
18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
20. HISTOLOGY: no, yes 
'''

import numpy as np
import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

''' READ & PREP DATA '''

col_names = ['status', 'age', 'sex', 'steriod', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 
'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']

data = pd.read_csv('hepatitis.data', na_values = ['?'], names = col_names, header = None)

print('Initial dataframe:\n', data)


''' KNN Imputation (missing values replacement) '''

col_names = data.columns

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=5)
fill_data = imputer.fit_transform(data)
# print('Integer imputation:\n', fill_data)

# Convert to dataframe
fill_data = pd.DataFrame(fill_data, columns=col_names)
# print('Integer imputation converted to dataframe:\n', fill_data)

# Create dataframe with integer values
col_int = fill_data.drop(['bilirubin', 'albumin'], axis = 1)

# Create dataframe with non-integer values
col_bilirubin = fill_data.bilirubin
col_albumin = fill_data.albumin
col_float = pd.concat([col_bilirubin, col_albumin], axis=1)
for col in col_float.columns:
    col_float[col] = col_float[col].astype('float32')
# print('Col float:\n', col_float)

# Round int cols to nearest integer
col_int = np.round(col_int)
for col in col_int.columns:
    col_int[col] = col_int[col].astype('int32')
# print('Int rounded:\n', col_int)

# Concatenate frames
data = pd.concat([col_int, col_float], axis=1)



# data.to_csv('hepatitisCleaned.csv')

# data = pd.read_csv('hepatitisCleaned.csv', index_col=0)
# print('Initial dataframe:\n', data)



''' Create Testing and Training sets '''

# Create dataframe for die
die_df = data[data['status'] == 1]

# Create dataframe for live
live_df = data[data['status'] == 2]

# Splitting dataframe into training and testing with equal ratio of die and live in each

xDie = die_df.drop(['status'], axis = 1)
yDie = die_df['status']

# Split Training set (67%) and Testing set (33%)
xDie_train, xDie_test, yDie_train, yDie_test = train_test_split(xDie, yDie, test_size = 0.33)

xLive = live_df.drop(['status'], axis = 1)
yLive = live_df['status']

# Split Training set (67%) and Testing set (33%)
xLive_train, xLive_test, yLive_train, yLive_test = train_test_split(xLive, yLive, test_size = 0.33)



# Combine samples to create testing and training data

x_train = pd.concat([xLive_train, xDie_train])
y_train = pd.concat([yLive_train, yDie_train])
x_test = pd.concat([xLive_test, xDie_test])
y_test = pd.concat([yLive_test, yDie_test])


column_names = x_train.columns



# # Principal Component Analysis

# # Components created: 1. Linear Combinations of original attributes
# #                    2. Perpendicular to each other
# #                    3. Capture the maximum amount of variation in the data.
# from sklearn.decomposition import PCA

# # intialize pca and logistic regression model
# pca = PCA()

# # fit and transform data
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)



# Robust Scaler

scaler = preprocessing.RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = pd.DataFrame(x_train, columns=column_names)
x_test = pd.DataFrame(x_test, columns=column_names)



# Read out to CSV

x_train.to_csv('xtrainhepScaled.csv')
x_test.to_csv('xtesthepScaled.csv')
y_train.to_csv('ytrainhepScaled.csv')
y_test.to_csv('ytesthepScaled.csv')