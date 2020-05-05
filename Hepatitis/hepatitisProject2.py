'''
Project 2: Decision Tree
Develop model based on decision tree
Learn exactly how it works and apply
Approximately 100 records for learning and 50 for testing

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree


''' READ IN DATA '''

x_train = pd.read_csv('xtrainhep88.csv', index_col=0)
x_test = pd.read_csv('xtesthep88.csv', index_col=0)
y_train = pd.read_csv('ytrainhep88.csv', index_col=0)
y_test = pd.read_csv('ytesthep88.csv', index_col=0)

column_names = x_train.columns


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
classifier = classifier.fit(x_train, y_train)

# Display Tree
# This outputs a txt file which can be copy and pasted into http://webgraphviz.com/ to display the tree
with open("hepatitis_gini.txt", "w") as f:
    f = tree.export_graphviz(classifier, feature_names = column_names, class_names = ['die','live'], filled = True, out_file=f)



# Test tree model

#Predict Test Data
y_predictTest = classifier.predict(x_test)
y_predictTrain = classifier.predict(x_train)

# Calculate accuracy
# Accuracy
print("DT Train Accuracy: ", accuracy_score(y_train, y_predictTrain)*100, "\n")
print("DT Test Accuracy: ", accuracy_score(y_test, y_predictTest)*100, "\n")

# Create Confuion Matrix
print("DT Confusion Matrix:\n", confusion_matrix(y_test, y_predictTest))

# Compute Report
print("DT Report:\n" , classification_report(y_test, y_predictTest))
