#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:45:39 2020

@author: jamie
"""
#TODO: apply weight to data
import pandas as pd
import numpy as np

#Read in data from csv file. 'Infer' means that the header is the default of the first row for each column
data = pd.read_csv('DCYA2018.csv')

data = pd.DataFrame(data)

#Drop Empty columns
#'DontFitIn', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo'
data = data.drop(['OtherProblems', 'Homeless2015', 'DateSexRecode'], axis=1)

print(data.head())
data.columns
#data.iloc[ : , 6 ]
#data.astype({'Race': 'int32'}).dtypes


# In[2]:


#Find Missing Values
data = data.replace(' ', np.NaN)
#data = data.drop(['DateSexRecode'], axis=1)

print(data.shape)
print('Numer of instances = %d' %data.shape[0])
print('Numer of attributes = %d' %data.shape[1])
print('Number of Missing Values')
data = data.fillna(data.median())

for col in data.columns:
    #count number of missing values in each column
    if data[col].isna().sum() > 0:
        print('\t%s: %d' %(col, data[col].isna().sum()))
    


# In[3]:


#Dimensionality of DataFrame
print("Number of rows: ", data.shape[0])
print("Number of columns: ",data.shape[1], '\n')

#Find columns where the variable has 64 bit signature (this is too large)
for col in data.columns:
    if data.dtypes[col] == 'int64' or data.dtypes[col] == 'float64':
        print(col, data.dtypes[col])

#Change these to 32 bits
#data['Schoolcode'] = data.Year.astype('float32')
#data['RespondentID'] = data.Year.astype('int32')
#data['Weighting2'] = data.Year.astype('float32')
#data['EverTestSTI'] = data.RespondentID.astype('int32')
#data['DateSexRecode'] = data.Year.astype('float32')

for col in data.columns:
    if data.dtypes[col] == 'int64':
        data[col] = data[col].astype('float32')
    if data.dtypes[col] == 'float64' or data.dtypes[col] =='object':
        data[col] = data[col].astype('float32')
data = data.astype('float32')
print(data.dtypes)

#Find columns where the variable has 64 bit signature (this is too large)
print("\nNot in requirement:\n")
for col in data.columns:
    if data.dtypes[col] != 'int32' and data.dtypes[col] != 'float32' and data.dtypes[col] != 'object':
        print(col, data.dtypes[col])
        
data.dtypes['IBelong']
data['IBelong'].value_counts()


# In[4]:


#Find Number of each value
print("Number of each value\n",data['IBelong'].value_counts())

#Create 2 classes: those who feel like they belong and those who do not
belong_map = {1.0:1,
              2.0:1,
              3.0:2,
              4.0:2}
                 
data["belong"] = data["IBelong"].map(belong_map)
data['belong'] = data.belong.astype('float32')

print("\nNumber of each value (belong):\n")
print(data['belong'].value_counts())

#Display original column (IBelong) compared to the new 2-variable column (belong)
print("Original Column: \n")
print(data[pd.notnull(data['IBelong'])]['IBelong'])

print("\nNew Column: \n")
print(data[pd.notnull(data['belong'])]['belong'])

data[['IBelong', 'belong']] 
#


# In[5]:


#Create table for Strongly Agree
SA = data[data['IBelong'] == 1.0]

#Create table for Agree
A = data[data['IBelong'] == 2.0]

#Create table for Disagree
D = data[data['IBelong'] == 3.0]

#Create table for Strongly Disagree
SD = data[data['IBelong'] == 4.0]


# In[6]:


#Create sample for training data by taking 66% of each category to get predictor and prediction classes
import sklearn
from sklearn.model_selection import train_test_split

    #Strongly Agree Represenation
#Value you are looking for
ySA = SA['belong']
#Dataset with missing value
xSA = SA.drop(['Schoolcode','IBelong','belong'],axis=1)

#Split Training set (67%) and Testing set (33%)
xSA_train,xSA_test,ySA_train,ySA_test=train_test_split(xSA,ySA,test_size=0.33)
#Check for right amount of rows- it is!
print("Total SA: ", xSA_test.shape[0] + xSA_train.shape[0], "\n  SA Test: ",xSA_test.shape[0],"\n  SA Train: ",xSA_train.shape[0])

    #Agree Represenation
#Value you are looking for
yA = A['belong']
#Dataset with missing value
xA = A.drop(['Schoolcode','IBelong','belong'],axis=1)

#Split Training set (67%) and Testing set (33%)
xA_train,xA_test,yA_train,yA_test=train_test_split(xA,yA,test_size=0.33)
print("\nTotal A: ", xA_test.shape[0] + xA_train.shape[0], "\n  A Test: ",xA_test.shape[0],"\n  A Train: ",xA_train.shape[0])

    #Disagree Represenation
#Value you are looking for
yD = D['belong']
#Dataset with missing value
xD = D.drop(['Schoolcode','IBelong','belong'],axis=1)

#Split Training set (67%) and Testing set (33%)
xD_train,xD_test,yD_train,yD_test=train_test_split(xD,yD,test_size=0.33)
print("\nTotal D: ", xD_test.shape[0] + xD_train.shape[0], "\n  D Test: ",xD_test.shape[0],"\n  D Train: ",xD_train.shape[0])

    #Strongly Disagree Represenation
#Value you are looking for
ySD = SD['belong']
#Dataset with missing value
xSD = SD.drop(['Schoolcode','IBelong','belong'],axis=1)

#Split Training set (67%) and Testing set (33%)
xSD_train,xSD_test,ySD_train,ySD_test=train_test_split(xSD,ySD,test_size=0.33)
print("\nTotal SD: ", xSD_test.shape[0] + xSD_train.shape[0], "\n  SD Test: ",xSD_test.shape[0],"\n  SD Train: ",xSD_train.shape[0])


# In[7]:


#Create training data: combine samples for your X and Y

#Create variable holding all training data sets with data being classified
trainFrames = [xSA_train, xA_train, xD_train, xSD_train]
#Combine frames to make training sample
x_train = pd.concat(trainFrames)

#Create variable holding all training data sets for predicted data
ytrainFrames = [ySA_train, yA_train, yD_train, ySD_train]
#Combine frames to make training sample
y_train = pd.concat(ytrainFrames)


# In[8]:


#Create Testing Data: combine samples for your X and Y 

#Create variable holding all testing data sets with data being classified
testFrames = [xSA_test, xA_test, xD_test, xSD_test]
#Combine frames to make testing sample
x_test = pd.concat(testFrames)

#Create variable holding all testing data sets with data being classified
ytestFrames = [ySA_test, yA_test, yD_test, ySD_test]
#Combine frames to make testing sample
y_test = pd.concat(ytestFrames)


# In[9]:


#Check if data is split correctly

print("Number of Rows in Training Sample (X): ", x_train.shape[0])
print("Number of Rows in Training Sample (Y): ", y_train.shape[0]) #they match!

print("\nNumber of Rows in Testing Sample (X): ", x_test.shape[0])
print("Number of Rows in Testing Sample (Y): ", y_test.shape[0]) #these match too!

#test if you have the right number of rows:
print("\nTotal Data: ", y_test.shape[0]+y_train.shape[0])

#Result: x_train, y_train, x_test, and y_test data sets


# In[229]:


#Principal Component Analysis
#Components created: 1. Linear Combinations of original attributes
#                    2. Perpendicular to each other
#                    3. Capture the maximum amount of variation in the data.
from sklearn.decomposition import PCA

#training = [x_train]
#training = pd.concat(training)
#training = training.astype('float32')

for col in x_train.columns:
    if x_train.dtypes[col] == 'int64':
        x_train[col] = x_train[col].astype('float32')
    if x_train.dtypes[col] == 'float64' or x_train.dtypes[col] =='object':
        x_train[col] = x_train[col].astype('float32')
x_train = x_train.astype('float32')
#x_train = x_train.drop([0], axis=1)
print(x_train.dtypes)

x_train = x_train.fillna(data.median())
print(x_train.shape)

numComponents = 32
pca = PCA(n_components=numComponents)
pca.fit(x_train)

projectedTrain = pca.transform(x_train)
print(projectedTrain.shape)

projectedTest = pca.transform(x_test)
print(projectedTest.shape)

projectedTrain = pd.DataFrame(projectedTrain,index=range(0,10910))
projectedTrain['belong'] = data['belong']
 
projectedTest = pd.DataFrame(projectedTest,index=range(0,5378))
projectedTest['belong'] = data['belong']
 
projectedTrain
projectedTest


# In[230]:


#Create Decision Tree PROJECTED
from sklearn import tree
#clf stores the model
clfProj = tree.DecisionTreeClassifier(criterion='gini',random_state = 100, max_depth=6)#, max_leaf_nodes=19)
#fit the data you are using to train
clfProj = clf.fit(projectedTrain, y_train)

Y_predTrain = clfProj.predict(projectedTrain)
Y_predTest = clfProj.predict(projectedTest)
    
print("Projected Train Accuracy:", accuracy_score(y_train,Y_predTrain)*100, "\n")
print("Projected Test Accuracy:", accuracy_score(y_test,Y_predTest)*100, "\n")


##
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report 
import matplotlib.pyplot as plt

#Model Fitting and Evaluation
maxdepths = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]

trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

index=0
for depth in maxdepths:
    clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=depth)
    clf = clf.fit(projectedTrain, y_train)
    
    Y_predTrain = clf.predict(projectedTrain)
    Y_predTest = clf.predict(projectedTest)
    
    trainAcc[index] = accuracy_score(y_train, Y_predTrain)
    testAcc[index] = accuracy_score(y_test, Y_predTest)
    print('Depth:', depth, 'Train:', trainAcc[index], 'Test:', testAcc[index])
    index += 1


# In[228]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1.0, 2.0]
colors = ['g', 'r',]
for target, color in zip(targets,colors):
    indicesToKeep = projected['belong'] == target
    ax.scatter(
                projected.loc[indicesToKeep, 'pc3']
               , projected.loc[indicesToKeep, 'pc1']
               #, projected.loc[indicesToKeep, 'pc1']
               , c = color
               #, marker=belongMarker[belongType]
               )
ax.legend(targets)
ax.grid()


# In[ ]:


#Display Projeted Values
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 3', fontsize = 15)
ax.set_zlabel('Principal Component 1', fontsize = 15)

plotBelong = projected[projected['belong']==1.0]
plotNBelong = projected[projected['belong']==2.0]

belongColor = {1.0:'g', 2.0:'r'}
belongMarker = {1.0:'+', 2.0:'o'}

for belongType in belongMarker:
    d = projected[projected['belong']==belongType]
    ax.scatter(d['pc2'],d['pc3'],d['pc1'],c=belongColor[belongType],marker=belongMarker[belongType])


# In[232]:


from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report 
import matplotlib.pyplot as plt

#Model Fitting and Evaluation
maxdepths = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]

trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

index=0
for depth in maxdepths:
    clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=depth)
    clf = clf.fit(x_train, y_train)
    Y_predTrain = clf.predict(x_train)
    Y_predTest = clf.predict(x_test)
    trainAcc[index] = accuracy_score(y_train, Y_predTrain)
    testAcc[index] = accuracy_score(y_test, Y_predTest)
    print('Depth:', depth, 'Train:', trainAcc[index], 'Test:', testAcc[index])
    index += 1
    
plt.plot(maxdepths,trainAcc,'ro-',maxdepths,testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')


# In[233]:


#Pre Pruning cont.
maxLeaves = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,35,40]
index=0
trainAcc = np.zeros(len(maxLeaves))
testAcc = np.zeros(len(maxLeaves))

for leaf in maxLeaves:
    clf = tree.DecisionTreeClassifier(max_depth=5,max_leaf_nodes=leaf)
    clf = clf.fit(x_train, y_train)
    Y_predTrain = clf.predict(x_train)
    Y_predTest = clf.predict(x_test)
    trainAcc[index] = accuracy_score(y_train, Y_predTrain)
    testAcc[index] = accuracy_score(y_test, Y_predTest)
    print('Leaf Nodes:', leaf, 'Train:', trainAcc[index], 'Test:', testAcc[index])
    index += 1
    


# In[235]:


#Create Decision Tree
from sklearn import tree
#clf stores the model
clf = tree.DecisionTreeClassifier(criterion='gini',random_state = 100, max_depth=5, max_leaf_nodes=19)
#fit the data you are using to train
clf = clf.fit(x_train, y_train)
#Create variables 
#Y = data['belong']
#X = data.drop(['Schoolcode', 'RespondentID','IBelong','belong'],axis=1)


# In[236]:


#Display Tree!
import pydotplus
from IPython.display import Image

dot_data = tree.export_graphviz(clf, feature_names=x_train.columns, class_names=['belong','not belong'], filled=True, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())


# In[220]:


#Predict Test Data
y_predictTest = clf.predict(x_test)
y_predictTrain = clf.predict(x_train)

from sklearn.metrics import accuracy_score, classification_report 
print("Train Accuracy:", accuracy_score(y_train,y_predictTrain)*100, "\n")
print("Test Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")


#Compute Report
#print("Report: \n" , classification_report(y_test, y_predictTest))


# In[166]:


#Create Confuion Matrix

from sklearn.metrics import confusion_matrix 
print("Decision Tree Confusion Matrix:")
confusion_matrix(y_test, y_predictTest)


# In[18]:


#Compute Accuracy
from sklearn.metrics import accuracy_score, classification_report 

print("Accuracy:", accuracy_score(y_test,y_predictTest)*100, "\n")


# In[ ]:


#Compute Report
print("Report: \n" ,
classification_report(y_test, y_predictTest))


# In[ ]:


import matplotlib.pyplot as plt

#Plot training and testing accuracies
plt.plot(maxdepths,trainAcc,'ro-',maxdepths,testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')


# In[ ]:


#Try again with criterion as entropy

clfE = tree.DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3)
#fit the data you are using to train
clfE = clfE.fit(x_train, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['Schoolcode', 'RespondentID','IBelong','belong'],axis=1)

#Display Tree
dot_dataE = tree.export_graphviz(clfE, feature_names=x_train.columns, class_names=['1','2'], filled=True, out_file=None) 
graphE = pydotplus.graph_from_dot_data(dot_dataE) 
Image(graphE.create_png())


# In[ ]:


#Entropy cont. 

#Predict Test Data
y_predictE = clfE.predict(x_test)
#Create Confuion Matrix
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix:")
confusion_matrix(y_test, y_predict)
from sklearn.metrics import accuracy_score, classification_report 
print("Accuracy:", accuracy_score(y_test,y_predictE)*100, "\n") #ok so it's a tiny bit more accurate... is it worth the change?

#Compute Report
print("Report: \n" , classification_report(y_test, y_predictE))


# In[ ]:


#Create Confuion Matrix
print("Entropy Confusion Matrix:")
confusion_matrix(y_test, y_predictE)
#When we apply cost, we will see which distribution is less harmful


# In[ ]:


#TODO: add cost, add weight, add comments on each line (know dif. between entropy and gini),precision,recall, f1-score, support


# In[ ]:


#Support Vector Machine
from sklearn.svm import SVC
svcclassifier = SVC(kernel='linear')
svcclassifier.fit(x_train, y_train)

y_predictSVM = svcclassifier.predict(x_test)


# In[ ]:



print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predictSVM))

#Compute Accuracy
print("Accuracy:", accuracy_score(y_test,y_predictSVM)*100, "\n")

#Compute Report
print("Report: \n" , classification_report(y_test, y_predictSVM))

