import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

# Read in data
irisData = pd.read_csv('iris.data')
irisData.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
print(irisData)

# Separate data into three vectors, one for each flower class
setosa = irisData[irisData['class'] == 'Iris-setosa'].drop(columns='class')
print("\nSetosa")
print(setosa)
versicolor = irisData[irisData['class'] == 'Iris-versicolor'].drop(columns='class')
print("\nVersicolor")
print(versicolor)
virginica = irisData[irisData['class'] == 'Iris-virginica'].drop(columns='class')
print("\nVirginica")
print(virginica)

# Calculate the means
setosa_mean = setosa.mean(axis = 0)
print("\nSetosa mean values:")
print(setosa_mean)
versicolor_mean = versicolor.mean(axis = 0)
print("\nVersicolor mean values:")
print(versicolor_mean)
virginica_mean = virginica.mean()
print("\nVirginica mean values:")
print(virginica_mean)

# Calculate the standard deviations
setosa_stdev = np.std(setosa)
print("\nSetosa standard deviation values:")
print(setosa_stdev)
versicolor_stdev = np.std(versicolor)
print("\nVersicolor standard deviation values:")
print(versicolor_stdev)
virginica_stdev = np.std(virginica)
print("\nVirginica standard deviation values:")
print(virginica_stdev)

# Euclidean distance as an array
setosa_dist = np.linalg.norm(setosa - setosa_mean, axis=1)
print("\nSetosa Euclidean distance as array:")
print(setosa_dist)
versicolor_dist = np.linalg.norm(versicolor - versicolor_mean, axis=1)
print("\nVersicolor Euclidean distance as array:")
print(versicolor_dist)
virginica_dist = np.linalg.norm(virginica - virginica_mean, axis=1)
print("\nVirginica Euclidean distance as array:")
print(virginica_dist)

# Euclidean distance by record
dist_setosa = np.sqrt(((setosa - setosa_mean)**2).sum(axis=1))
print("\nSetosa Euclidean distance by record:")
print(dist_setosa)

dist_versicolor = np.sqrt(((versicolor - versicolor_mean)**2).sum(axis=1))
print("\nVersicolor Euclidean distance by record:")
print(dist_versicolor)

dist_virginica = np.sqrt(((virginica - virginica_mean)**2).sum(axis=1))
print("\nVirginica Euclidean distance by record:")
print(dist_virginica)

# Basic x-y plots
plt.plot(irisData['sepal length'], irisData['sepal width'], 'o')

fig, axes = plt.subplots(3, 2, figsize=(12,12))
index = 0
for i in range(3):
    for j in range(i+1,4):
        ax1 = int(index/2)
        ax2 = index % 2
        axes[ax1][ax2].scatter(irisData[irisData.columns[i]], irisData[irisData.columns[j]], color='red')
        axes[ax1][ax2].set_xlabel(irisData.columns[i])
        axes[ax1][ax2].set_ylabel(irisData.columns[j])
        index = index + 1


