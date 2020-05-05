'''
Project 4
Find the best ways to explore, preprocess, and visualize the data. 
Compare between the performance of the Decision Tree and SVM.

Hepatitis Data set
155 records | 19 attributes + 1 for variable live or die

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


# ''' READ & PREP DATA '''

data = pd.read_csv('hepatitisCleaned.csv', index_col=0)
# print('Initial dataframe:\n', data)

dt_accuracy_array = []
svm_accuracy_array = []
num_loops = 100

for x in range(num_loops):

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


    # Splitting dataframe into training and testing without making sure equal ratio
    #Dataset with missing value
    x = data.drop(['status'], axis = 1)
    #Value you are looking for
    y = data['status']
    #Split Training set (67%) and Testing set (33%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)


    column_names = x_train.columns


    # Robust Scaler
    scaler = preprocessing.RobustScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=column_names)




    ''' Decision Tree '''

    # Create Decision Tree

    # classifier = tree.DecisionTreeClassifier(criterion='gini',max_depth=4, max_leaf_nodes=6, min_samples_leaf=5, min_samples_split=3)
    classifier = tree.DecisionTreeClassifier(criterion='gini',max_depth=3, max_leaf_nodes=3, min_samples_leaf=3, min_samples_split=2)

    # Perform training
    classifier = classifier.fit(x_train, y_train)


    # Test tree model

    # Predict Test Data
    y_predictTest = classifier.predict(x_test)

    # Calculate accuracy
    dt_accuracy = accuracy_score(y_test, y_predictTest)*100

    # Append to array
    dt_accuracy_array.append(dt_accuracy)


    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    ''' Support Vector Machine '''

    # Create SVM (using the support vector classifier class - SVC)

    svcclassifier = SVC(kernel='poly', C=0.1, gamma=0.1, degree=3, class_weight='balanced') #87.38 88.46

    svcclassifier.fit(x_train, y_train)


    # Test tree model

    #Predict Test Data
    y_predictTest = svcclassifier.predict(x_test)

    svm_accuracy = accuracy_score(y_test, y_predictTest)*100

    # Append to array
    svm_accuracy_array.append(svm_accuracy)



dt_string = [("%.2f%%" % a) for a in dt_accuracy_array]
dt_string = [str(a) for a in dt_string]

svm_string = [("%.2f%%" % a) for a in svm_accuracy_array]
svm_string = [str(a) for a in svm_string]

print(num_loops, 'Random Test/Train Splits\n')
print('DT Average Accuracy: ', "%.2f%%" % (sum(dt_accuracy_array)/num_loops))
print('SVM Average Accuracy: ', "%.2f%%" % (sum(svm_accuracy_array)/num_loops))

print('\nDT Max Accuracy: ', max(dt_string), '\nDT Min Accuracy: ', min(dt_string))
print('\nSVM Max Accuracy: ', max(svm_string), '\nSVM Min Accuracy: ', min(svm_string))

# print('DT Accuracy: ', ', '.join(dt_string))
# print('SVM Accuracy: ', ', '.join(svm_string))
