import numpy as np
import pandas as pd

# Read in data
heart_data = pd.read_csv('goodData.csv', na_values = ['-9'])

print('Initial dataframe:\n', heart_data)

# Drop columns 75-421
heart_data = heart_data.drop(heart_data.iloc[:, 75:421], axis = 1)

# Assign column names, 75 attributes since patient name has been removed
heart_data.columns = ['id', 'ssn', 'age', 'sex', 'painloc', 'painexer', 'relrest', 
'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 
'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyear', 'dig', 'prop', 'nitr', 'pro', 
'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 
'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 
'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev-', 
'thalpul-', 'earlobe-', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 
'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1_', 'lvx2_', 'lvx3_', 'lvx4_', 
'lvf_', 'cathef_', 'junk_']

# Print clean data
print('Cleaner data:\n', heart_data, '\n')

# # Get rid of columns with just one value
# std = np.std(heart_data)
# cols_to_drop = std[std==0].index
# heart_data = heart_data.drop(cols_to_drop, axis=1)
# print('Columns with a single value removed:\n', cols_to_drop, '\n')

# Print datatypes for each column
print('Datatypes for each column\n', heart_data.dtypes, '\n')

# Make a new data frame with just the identifying attributes
identifying_attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'cigs', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
test_data = heart_data[identifying_attributes]
print(test_data)

# Write out test data
test_data.to_csv('clevelandHeartDisease.csv', encoding='utf-8', index=False)
