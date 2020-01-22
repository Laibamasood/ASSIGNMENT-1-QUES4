# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:21:37 2020

@author: Laiba Masood
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Id vs sale (Training set)')
plt.xlabel('ID')
plt.ylabel('PRICE')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Id vs Sale (Test set)')
plt.xlabel('ID')
plt.ylabel('PRICE')
plt.show()