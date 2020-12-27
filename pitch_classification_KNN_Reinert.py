# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:15:40 2020

@author: reine
"""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


"""DATA PREPROCESSING"""
# Read Data File
print('Reading file...')
df = pd.read_csv('fb_os_df.csv', encoding='iso-8859-1')
df = df.drop(columns=['Unnamed: 0'])


# Split into Test/Train groups
print('Creating test/train groups...')
X = df.iloc[:, 0:-1] 
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
print('Scaling features...')

nn = 125

print(f'Creating knn classifier with {nn} nearest neighbors...')
knn = KNeighborsClassifier(n_neighbors = nn)
print('Training the model...')
knn.fit(X_train, y_train)

print('Predicting test dataset...')
y_pred = knn.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

