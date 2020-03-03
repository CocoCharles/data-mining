'''
Heart Disease Data Set
Source: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
Find the important attributes - look at the 14 they studied. 
'Num' is predicted attribute
'''

'''
Attributes:
#3 Age in years (age)
#4 Sex: 1=male, 0=female (sex)
#9 Chest pain type (cp)
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic
#10 Resting blood pressure (trestbps)
-- in mm Hg on admission to the hospital
#12 Serum cholesterol in mg/dl (chol)
#14 Cigarettes per day (cigs)
#16 Fasting blood sugar > 120 mg/dl: 1 = true; 0 = false (fbs)
#19 Resting electrocardiographic results (restecg)
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
#32 Maximum heart rate achieved (thalach)
#38  Exercise induced angina: 1 = yes; 0 = no (exang)
#40 Oldpeak = ST depression induced by exercise relative to rest (oldpeak)
#41 Slope of the peak exercise ST segment (slope)
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping
#44 Number of major vessels (0-3) colored by fluoroscopy (ca)
#51 Thal: 3 = normal; 6 = fixed defect; 7 = reversible defect (thal)
#58 Diagnosis of heart disease (angiographic disease status) (num) (the predicted attribute)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from pandas.api.types import is_numeric_dtype
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

# Read in data
heart_data = pd.read_csv('clevelandHeartDisease.csv')
print('Initial dataframe:\n', heart_data)

''' PREP DATA '''

# Number of NaN values in each column
print('Number of NaN values in each column:\n', heart_data.isnull().sum())

# Add a column that converts 'num' to only 0 v 1
# With 0 being no diagnosis and 1 diagnosed
# Replacing all values >=1 with
def diag_num (row):
    if row['num'] == 0:
        return '0'
    if row['num'] >= 1:
        return '1'

heart_data['num_diagnosis'] = heart_data.apply(lambda row: diag_num(row), axis=1)
print('Num column converted to binary instead of 0-4:\n', heart_data.num_diagnosis, '\n')

# Change datatype for num_diagnosis to int64
heart_data[['num_diagnosis']] = heart_data[['num_diagnosis']].astype('int64')

# Print datatypes for each column
print('Datatypes for each column\n', heart_data.dtypes, '\n')

''' STATS '''

# Statistical characteristics for each column
for col in heart_data.columns:
    if is_numeric_dtype(heart_data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % heart_data[col].mean())
        print('\t Median = %.2f' % heart_data[col].median())
        print('\t Standard deviation = %.2f' % heart_data[col].std())
        print('\t Minimum = %.2f' % heart_data[col].min())
        print('\t Maximum = %.2f' % heart_data[col].max())

# Separate data into two sets by 1 or 0 for num_diagnosis
pos_heart_disease = heart_data[heart_data['num_diagnosis'] == 1].drop(columns='num_diagnosis')
neg_heart_disease = heart_data[heart_data['num_diagnosis'] == 0].drop(columns='num_diagnosis')
# print('pos', pos_heart_disease)

# Number of cigs smoked by people with and withoug heart disease
print('Value counts for pos:\n', pos_heart_disease.cigs.value_counts())
print('Value counts for neg:\n', neg_heart_disease.cigs.value_counts())

# Print number of records in num_diagnosis that are either positive or negative
print('Value counts for num_diagnosis:\n', heart_data.num_diagnosis.value_counts())

''' GRAPHS '''

# Attributes to compare: 'age', 'trestbps', 'chol', 'cigs', 'restecg', 'ca', 'num_diagnosis'

# Creating two new data frames with continous attributes and discrete
continous_cols = ['age', 'trestbps', 'chol', 'num_diagnosis'] 
discrete_cols = ['cigs', 'restecg', 'ca', 'num_diagnosis']
compare_cont = heart_data[continous_cols]
compare_disc = heart_data[discrete_cols]

# PairGrid graph for continous comparing num_diagnos 0 (no heart disease) & 1 (heart disease)
g = sns.PairGrid(compare_cont, hue="num_diagnosis")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

# PairGrid graph for discrete, otherwise same as above
g = sns.PairGrid(compare_disc, hue="num_diagnosis")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

# Convert columns to compare to numpy
age_array = np.array(heart_data['age'])
bps_array = np.array(heart_data['trestbps'])
chol_array = np.array(heart_data['chol'])
cigs_array = np.array(heart_data['cigs'])
ecg_array = np.array(heart_data['restecg'])
ca_array = np.array(heart_data['ca'])
num_array = np.array(heart_data['num_diagnosis'])

# Function to make 3D histogram
def histogram_3D (xarray, yarray, xname, yname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(xarray, yarray, bins=(20,20))
    xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)

    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.title('Diagnosis Histogram')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

# 3D Histograms of age, trestbps, chol, cigs, restecg, ca vs. Diagnosis
histogram_3D(age_array, num_array, 'Age', 'Diagnosis')
histogram_3D(bps_array, num_array, 'Resting BPS', 'Diagnosis')
histogram_3D(chol_array, num_array, 'Cholesterol', 'Diagnosis')
# histogram_3D(cigs_array, num_array, 'Cigarettes per day', 'Diagnosis')
histogram_3D(ecg_array, num_array, 'Resting ECG', 'Diagnosis')
# histogram_3D(ca_array, num_array, 'Ca', 'Diagnosis')


# 3D scatter plots 
# age, trestbps, num_diagnosis
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(heart_data.age, heart_data.trestbps, heart_data.num_diagnosis, c=[0,1,0], s=100)
ax.legend()

# chol, cigs, num_diagnosis
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(heart_data.chol, heart_data.cigs, heart_data.num_diagnosis, c=[0,1,0], s=100)
ax.legend()

# restecg, ca, num_diagnosis
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(heart_data.restecg, heart_data.ca, heart_data.num_diagnosis, c=[0,1,0], s=100)
ax.legend()

# age, trestbps, chol split into pos & neg diagnosis
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pos_heart_disease.age, pos_heart_disease.trestbps, pos_heart_disease.chol, label='positive', c=[0,1,0], s=100)
ax.scatter(neg_heart_disease.age, neg_heart_disease.trestbps, neg_heart_disease.chol, label='negative', c=[1,0,0], s=100)
ax.legend()
plt.show()

# cigs, restecg, ca split into pos & neg diagnosis
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pos_heart_disease.cigs, pos_heart_disease.restecg, pos_heart_disease.ca, label='positive', c=[0,1,0], s=100)
ax.scatter(neg_heart_disease.cigs, neg_heart_disease.restecg, neg_heart_disease.ca, label='negative', c=[1,0,0], s=100)
ax.legend()
plt.show()


## 3D plot layout
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')

# xpos = 
# ypos = 
# zpos = np.zeros()

# dx =
# dy = 
# dz = 

# ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')
# plt.show()

# # DIFFERENT VERSION 3D Histogram comparing age_array and num_array
# fig = plt.figure()
# fig.suptitle('Age vs. Diagnosis Histogram')
# ax = fig.add_subplot(111, projection='3d')

# hist, xedges, yedges = np.histogram2d(age_array, num_array, bins=(6, 2), range = [[20, 80], [0,1]])
# xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

# xpos = xpos.flatten()/2.
# ypos = ypos.flatten()/2.
# zpos = np.zeros_like (xpos)

# dx = xedges [1] - xedges [0]
# dy = yedges [1] - yedges [0]
# dz = hist.flatten()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
# plt.xlabel ("Age")
# plt.ylabel ("Num Diagnosis")

# plt.show()

# # 2D Histogram Attempt
# fig = plt.figure(figsize=(12,12))
# fig.suptitle('Histogram with Titles?')
# ax = fig.add_subplot(111)
# axAge = fig.add_subplot(211)
# axBPS = fig.add_subplot(222)
# axChol = fig.add_subplot(313)
# axCigs = fig.add_subplot(324)

# axAge.hist(age_array, bins=12, range=[20, 80], color='b')
# axAge.set_title('Age')
# axBPS.hist(bps_array, bins=22, range=[90, 200], color='g')
# axBPS.set_title('Rest BPS')
# axChol.hist(chol_array, bins=45, range=[120, 570], color='y')
# axChol.set_title('Chol')
# axCigs.hist(cigs_array, bins=20, range=[0, 100], color='r')
# axCigs.set_title('Cigs/day')

# # 2D Histogram
# fig, ax = plt.subplots(3, 2, figsize=(12,12))
# fig.suptitle('2D Histograms')

# colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive', 'tab:yellow']
# ax[0][0].hist(age_array, bins=12, range=[20, 80], color='b', label='Age')
# ax[0][0].set_title('Age')
# ax[0][1].hist(bps_array, bins=22, range=[90, 200], color='g', label='Rest BPS')
# ax[0][1].set_title('Rest BPS')
# ax[1][0].hist(chol_array, bins=45, range=[120, 570], color='y', label='Chol')
# ax[1][0].set_title('Chol')
# ax[1][1].hist(cigs_array, bins=20, range=[0, 100], color='r', label='Cigs')
# ax[1][1].set_title('Cigs')
# ax[2][0].hist(ecg_array, bins=6, range=[0, 2], color='b', label='Rest ECG')
# ax[2][0].set_title('Rest ECG')
# ax[2][1].hist(ca_array, bins=8, range=[0, 3], color='y', label='Rest BPS')
# ax[2][1].set_title('Rest BPS')

# # 3D Histogram comparing age_array and num_array
# fig = plt.figure()
# fig.suptitle('Age vs. Diagnosis Histogram')
# ax = fig.add_subplot(111, projection='3d')

# hist, xedges, yedges = np.histogram2d(age_array, num_array, bins=(12, 4), range=[[20, 80], [0,1]])
# xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)

# xpos = xpos.flatten('F')
# ypos = ypos.flatten('F')
# zpos = np.zeros_like(xpos)

# dx = 0.5 * np.ones_like(zpos)
# dy = dx.copy()
# dz = hist.flatten()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
# plt.xlabel ('Age')
# plt.ylabel ('Diagnosis')

# # plt.show()