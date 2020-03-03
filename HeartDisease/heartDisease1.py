"""
Find the important attributes - look at the 14 they studied. 
'Num' is predicted attribute
300 records, each record has 75 attributes
200 records for training, 100 for test set
class is based on predicted attribute
create classification decision tree
if diameter is closed by 50% or more or less

choose one algorithm for making decision tree, 
learn exactly how it works and apply
look at python decision tree id3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Read in data
heart_data = pd.read_csv('clevelandHeartDisease.csv')

# Drop the first column and columns 75-421
heart_data.drop(heart_data.columns[0], inplace = True, axis=1)
heart_data.drop(heart_data.iloc[:, 75:421], inplace = True, axis = 1)

# Assign column names
heart_data.columns = ['id', 'ssn', 'age', 'sex', 'painloc', 'painexer', 'relrest', 
'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 
'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyear', 'dig', 'prop', 'nitr', 'pro', 
'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 
'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 
'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev-', 
'thalpul-', 'earlobe-', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 
'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1_', 'lvx2_', 'lvx3_', 'lvx4_', 
'lvf_', 'cathef_', 'junk_']

print('Initial dataframe:\n', heart_data)

# hdMatrix = []
# hdMatrix = heart_data
# print(hdMatrix)

# Get rid of columns with just one value
std = np.std(heart_data)
cols_to_drop = std[std==0].index
heart_data.drop(cols_to_drop, inplace = True, axis=1)
print('Columns with a single value removed:\n', cols_to_drop, '\n')

# # Write out new data
# # heart_data.to_csv('chd.csv', encoding='utf-8', index=False)

# Find columns with negative values
col_contain_neg = (heart_data >= 0).all(0).eq(False)
print('Columns with negative numbers:\n', col_contain_neg, '\n')
# cigs, years, dm, dig, prop, nitr, pro, diuretic, thaltime, ca, thal

# Replace negative nums to 0
heart_data = heart_data.clip(lower=0)

# Print clean data
print('Cleaner data:\n', heart_data, '\n')

# Print datatypes for each column
print('Datatypes for each column\n', heart_data.dtypes, '\n')

# Change boolean columns to dtype bool
# sex, painloc_, painexer_, relrest_, smoke_, fbs, dm, famhist, dig, prop, nitr, pro, diuretic, exang, xhypo
bool_cols = []
for col in heart_data.columns:
    unique = heart_data[col].unique()
    if (len(unique) == 2 and (0 in unique) and (1 in unique)):
        bool_cols.append(col)
print('Boolean columns: ', bool_cols, '\n')

for col in bool_cols:
    heart_data[col] = heart_data[col].astype('bool')
print(heart_data[bool_cols], '\n')

# Print datatypes for each column
print('Datatypes for each column\n', heart_data.dtypes, '\n')

identifying_attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
test_data = heart_data[identifying_attributes]
print(test_data)

# # Write out new data
# test_data.to_csv('oldCleanHDCompare.csv', encoding='utf-8', index=False)

 # Calculate percent female and male
sexCount = Counter(heart_data.sex)
print(sexCount)

percentFem = (sexCount[0] / len(heart_data.axes[0]))
print("Percent female: {:.2%}".format(percentFem))

percentMale = (sexCount[1] / len(heart_data.axes[0]))
print("Percent male: {:.2%}".format(percentMale))

# Create 3D plots
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
fig1, ax1 = plt.subplots()

ax.scatter(heart_data.trestbps, heart_data.chol, label="Age", c=[1,0,1], s=100)
ax.scatter(heart_data.thalach, heart_data.tpeakbpd, label="Sex", c=[0,0,1], s=100)
ax.scatter(heart_data.thalrest, heart_data.tpeakbps, label="Cigs", c=[1,0,0], s=100)
ax.legend()

ax1.plot(heart_data['trestbps'], heart_data['chol'], 'o')
plt.show()


#age trestbps chol (cigs years) thalach thalrest tpeakbps tpeakbpd trestbpd oldpeak rldv5e
compare_cols = heart_data[['trestbps', 'tpeakbps', 'tpeakbpd', 'trestbpd']]
print(compare_cols)

fig, axes = plt.subplots(3, 2, figsize=(12,12))
index = 0
for i in range(3):
    for j in range(i+1,4):
        ax1 = int(index/2)
        ax2 = index % 2
        axes[ax1][ax2].scatter(compare_cols[compare_cols.columns[i]], compare_cols[compare_cols.columns[j]], color='red')
        axes[ax1][ax2].set_xlabel(compare_cols.columns[i])
        axes[ax1][ax2].set_ylabel(compare_cols.columns[j])
        index = index + 1
plt.show()
        