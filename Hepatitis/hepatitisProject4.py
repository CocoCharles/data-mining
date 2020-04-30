'''
Project 4
Finding the best way to explore, preprocess, and visualize the data
Compare methods used in projects 2 and 3

Hepatitis Data set
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
from sklearn.svm import SVC


''' READ & PREP DATA '''

col_names = ['status', 'age', 'sex', 'steriod', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 
'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']

data = pd.read_csv('hepatitis.data', na_values = ['?'], names = col_names, header = None)


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
print('Completed imputation:\n', data)



# data.to_csv('hepatitisCleaned.csv')
# data = pd.read_csv('hepatitisCleaned.csv', index_col=0)
# print('Initial dataframe:\n', data)




''' Creating Testing and Training sets '''

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


# # Splitting dataframe into training and testing without making sure equal ratio
# #Dataset with missing value
# x = data.drop(['status'], axis = 1)
# #Value you are looking for
# y = data['status']
# #Split Training set (67%) and Testing set (33%)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)


# x_train.to_csv('xtrainhep90.csv')
# x_test.to_csv('xtesthep90.csv')
# y_train.to_csv('ytrainhep90.csv')
# y_test.to_csv('ytesthep90.csv')

# x_train = pd.read_csv('xtrainhep90.csv', index_col=0)
# x_test = pd.read_csv('xtesthep90.csv', index_col=0)
# y_train = pd.read_csv('ytrainhep90.csv', index_col=0)
# y_test = pd.read_csv('ytesthep90.csv', index_col=0)


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



''' Decision Tree '''


# Grid Search - find best combination of parameters

# Parameter Grid
param_grid = {'criterion':['gini','entropy'], 'max_depth':[3,4,5,6], 'min_samples_leaf':[2,3,4,5], 'max_leaf_nodes':[3,4,5,6,7], 'min_samples_split':[2,3,4]}

# Make grid search classifier
clf_grid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, scoring='accuracy')

# Train the classifier
clf_grid.fit(x_test, y_test)

print("Best Parameters:\n", clf_grid.best_params_)
print('score\n', clf_grid.score(x_train, y_train))



# Create Decision Tree

classifier = tree.DecisionTreeClassifier(criterion='gini',max_depth=4, max_leaf_nodes=6, min_samples_leaf=5, min_samples_split=3)

# Perform training
classifier = classifier.fit(x_train, y_train)

# Display Tree!
with open("hepatitis_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = column_names, class_names = ['die','live'], filled = True, out_file=f)


# Test tree model

# Predict Test Data
y_predictTest = classifier.predict(x_test)
y_predictTrain = classifier.predict(x_train)

# Calculate accuracy
print("DT Train Accuracy: ", accuracy_score(y_train, y_predictTrain)*100, "\n")
print("DT Test Accuracy: ", accuracy_score(y_test, y_predictTest)*100, "\n")

# Create Confuion Matrix
print("DT Confusion Matrix:\n", confusion_matrix(y_test, y_predictTest))

# Compute Report
print("DT Report:\n" , classification_report(y_test, y_predictTest))


# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

''' Support Vector Machine '''

# Grid Search - find best combination of parameters

# Parameter Grid
param_grid = {'kernel': ['poly', 'rbf'], 'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'degree': [2,3,4], 'class_weight': ['balanced', None]}

# Make grid search classifier
clf_grid = GridSearchCV(SVC(), param_grid, scoring='accuracy')

# Train the classifier
clf_grid.fit(x_test, y_test)

print("Best Parameters:\n", clf_grid.best_params_)
print('score\n', clf_grid.score(x_train, y_train))


# Create SVM (using the support vector classifier class - SVC)

svcclassifier = SVC(kernel='poly', C=0.1, gamma=0.1, degree=3, class_weight='balanced') #87.38 88.46

svcclassifier.fit(x_train, y_train)


# Test tree model

#Predict Test Data
y_predictTest = svcclassifier.predict(x_test)
y_predictTrain = svcclassifier.predict(x_train)

print("SVM Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("SVM Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")

#Create Confuion Matrix
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_predictTest))

#Compute Report
# print("SVM Report: \n" , classification_report(y_test, y_predictTest))
