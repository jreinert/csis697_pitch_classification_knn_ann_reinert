# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:38:39 2020

@author: reine
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:30:12 2020

@author: reine
"""
# Functions
def adj_r2(r2, n, p):
    return 1 - (((1-r2)*(n-1))/(n-p-1))

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

"""DATA PREPROCESSING"""
# Read Data File
print('Reading file...')
df = pd.read_csv('fb_os_df.csv', encoding='iso-8859-1')
df = df.drop(columns=['Unnamed: 0'])


# Split into Test/Train groups
print('Creating test/train groups...')
X = df.iloc[:, 0:-1] 
y = df.iloc[:, -1].values.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
print('Scaling features...')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""Neural Network Model"""
print('Neural Networking Running....')
act_func='softplus'
num_neurons = 25
model = Sequential()
model.add(Dense(num_neurons, input_dim=X.columns.size, activation=act_func))
model.add(Dense(num_neurons, activation=act_func))
model.add(Dense(num_neurons, activation=act_func))
model.add(Dense(num_neurons, activation=act_func))
model.add(Dense(num_neurons, activation=act_func))
model.add(Dense(3, activation='softmax')) # <--- Output Layer
#model.summary()

#Train the Model
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=25)

y_predict = model.predict(X_test)

#Plot Loss Values
plt.plot(history.history['loss'])
plt.ylabel('Losses')
plt.xlabel('Epoch Number')

# CALCULATE R2 VALUE
from sklearn.metrics import r2_score
# Vars for Adj. r2 calculation
n = len(X_test)
p = len(X.columns)
r_squared = r2_score(y_test, y_predict)
adjusted_rsquared = adj_r2(r_squared, n, p)
print('r squared: ', r_squared)
print('adjusted r squared', adjusted_rsquared)

# Making the Confusion Matrix
print('\nCONFUSION MATRIX')
cm = confusion_matrix(y_test.argmax(axis = 1), y_predict.argmax(axis = 1))
print(f'\n{cm}')

total_right = cm[0][0] + cm[1][1] + cm[2][2]
total_wrong = cm[0][1] + cm[0][2] + cm[1][0] + cm[1][2] + cm[2][0] + cm[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the model is {total_acc * 100:.2f}%')



