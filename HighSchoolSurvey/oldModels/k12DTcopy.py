#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[4]:


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
data = pd.read_csv('DCYA2018.csv', na_values=[' '])

data = pd.DataFrame(data)
print(data.head())
print(data.columns)
#print(data.iloc[:,6])


# In[5]:


#Find Missing Values
data = data.replace(' ', np.NaN)
data = data.drop(['OtherProblems'], axis=1)

print(data.shape)
print('Numer of instances = %d' %data.shape[0])
print('Numer of attributes = %d' %data.shape[1])
print('Number of Missing Values')
data = data.fillna(data.median())

for col in data.columns:
    #count number of missing values in each column
    print('\t%s: %d' %(col, data[col].isna().sum()))
    


# In[6]:


#all column names
for col_name in data.columns: 
    print(col_name)
    #print("This: ",col_name if isinstance(type(col_name), object) else " none")


# In[7]:


#Dimensionality of DataFrame
print("Number of rows: ", data.shape[0])
print("Number of columns: ",data.shape[1], '\n')

#Find columns where the variable has 64 bit signature and change to 32 bits
for col in data.columns:
    if data.dtypes[col] == 'int64':
        data[col] = data[col].astype('int32')
    if data.dtypes[col] == 'float64':
        data[col] = data[col].astype('float32')

print(data.dtypes)


# In[8]:


#TODO: DO I NEED THIS? OR NAH- the next cell prob does the same thing and better

#Create variable grouping those who agree and strongly agree (feel like they belong)
agree = data[data['IBelong'].isin(['1', '2'])]

#Create variable grouping those who disagree and strongly disagree (feel like they do not belong)
notAgree = data[data['IBelong'].isin(['3', '4'])]


# In[9]:


#Create 2 classes: those who feel like they belong and those who do not
belong_map = {"4" : "2",
              "1" : "1", 
              "2" : "1", 
              "3" : "2"
              }
data['belong'] = data['IBelong'].map(belong_map)    

#Display original column (IBelong) compared to the new 2-variable column (belong)
print("Both Columns: \n")
print(data[pd.notnull(data['IBelong'])]['IBelong'], '\n', data[pd.notnull(data['belong'])]['belong'])


# In[10]:


#Create table for Strongly Agree
SA = data[data['IBelong'] == '1']

#Create table for Agree
A = data[data['IBelong'] == '2']

#Create table for Disagree
D = data[data['IBelong'] == '3']

#Create table for Strongly Disagree
SD = data[data['IBelong'] == '4']


# In[11]:


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


# In[12]:


#Create training data: combine samples for your X and Y

#Create variable holding all training data sets with data being classified
trainFrames = [xSA_train, xA_train, xD_train, xSD_train]
#Combine frames to make training sample
x_train = pd.concat(trainFrames)

#Create variable holding all training data sets for predicted data
ytrainFrames = [ySA_train, yA_train, yD_train, ySD_train]
#Combine frames to make training sample
y_train = pd.concat(ytrainFrames)


# In[13]:


#Create Testing Data: combine samples for your X and Y 

#Create variable holding all testing data sets with data being classified
testFrames = [xSA_test, xA_test, xD_test, xSD_test]
#Combine frames to make testing sample
x_test = pd.concat(testFrames)

#Create variable holding all testing data sets with data being classified
ytestFrames = [ySA_test, yA_test, yD_test, ySD_test]
#Combine frames to make testing sample
y_test = pd.concat(ytestFrames)


# In[14]:


#Check if data is split correctly

print("Number of Rows in Training Sample (X): ", x_train.shape[0])
print("Number of Rows in Training Sample (Y): ", y_train.shape[0]) #they match!

print("\nNumber of Rows in Testing Sample (X): ", x_test.shape[0])
print("Number of Rows in Testing Sample (Y): ", y_test.shape[0]) #these match too!

#test if you have the right number of rows:
print("\nTotal Data: ", y_test.shape[0]+y_train.shape[0])

#Result: x_train, y_train, x_test, and y_test data sets


# In[23]:


#Create Decision Tree
from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion='gini',random_state = 100,max_depth=3)
#fit the data you are using to train
classifier = classifier.fit(x_train, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['Schoolcode', 'RespondentID','IBelong','belong'],axis=1)


# In[24]:


#Display Tree!
import pydotplus
from IPython.display import Image

dot_data = tree.export_graphviz(classifier, feature_names=x_train.columns, class_names=['1','2'], filled=True, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())


# In[25]:


#Predict Test Data
y_predict = classifier.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report 
print("Accuracy:", accuracy_score(y_test,y_predict)*100, "\n")

#Compute Report
print("Report: \n" , classification_report(y_test, y_predict))


# In[26]:


#Create Confuion Matrix
from sklearn.metrics import confusion_matrix 
print("Gini Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))


# In[27]:


#Compute Accuracy
from sklearn.metrics import accuracy_score, classification_report 

print("Accuracy:", accuracy_score(y_test,y_predict)*100, "\n")


# In[28]:


#Compute Report
print("Report: \n" ,
classification_report(y_test, y_predict))


# In[29]:


#Try again with criterion as entropy

classifierE = tree.DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3)
#fit the data you are using to train
classifierE = classifierE.fit(x_train, y_train)
#Create variables 
Y = data['belong']
X = data.drop(['Schoolcode', 'RespondentID','IBelong','belong'],axis=1)

#Display Tree
dot_dataE = tree.export_graphviz(classifierE, feature_names=x_train.columns, class_names=['1','2'], filled=True, out_file=None) 
graphE = pydotplus.graph_from_dot_data(dot_dataE) 
Image(graphE.create_png())


# In[30]:


#Entropy cont. 

#Predict Test Data
y_predictE = classifierE.predict(x_test)
#Create Confuion Matrix
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix:")
confusion_matrix(y_test, y_predict)
from sklearn.metrics import accuracy_score, classification_report 
print("Accuracy:", accuracy_score(y_test,y_predictE)*100, "\n") #ok so it's a tiny bit more accurate... is it worth the change?

#Compute Report
print("Report: \n" , classification_report(y_test, y_predictE))


# In[31]:


#Create Confuion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predictE))
#When we apply cost, we will see which distribution is less harmful


# In[ ]:


#TODO: add cost, add weight, add comments on each line (know dif. between entropy and gini)

