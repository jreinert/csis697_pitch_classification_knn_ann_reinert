# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:30:12 2020

@author: reine
"""
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""DATA PREPROCESSING"""
# Read Data File
print('Reading file...')
df = pd.read_csv('pitches.csv', encoding='iso-8859-1')

# Create new df copy and drop null values and code, type other categorical columns
df2 = df
print('Dropping null data...')
df2 = df2.dropna()

print('Dropping unnecessary columns...')
df2 = df2.drop(columns=['code','type'])

# Drop pitch out and unknown pitch type data
df2 = df2[df2.pitch_type != 'FO']
df2 = df2[df2.pitch_type != 'IN']
df2 = df2[df2.pitch_type != 'PO']
df2 = df2[df2.pitch_type != 'UN']

print('Rewriting pitch types to FB and OS...')
# Update 'pitch_type' elements to new categories: FB, CH, BB
for i in df2.index:
    if df2.at[i,'pitch_type'] == 'FC' or df2.at[i,'pitch_type'] == 'FF' or df2.at[i,'pitch_type'] == 'FS' or df2.at[i,'pitch_type'] == 'FT':
        df2.at[i,'pitch_type'] = 'FB'
    elif df2.at[i,'pitch_type'] == 'CH':
        df2.at[i,'pitch_type'] = 'CH'
    else:
        df2.at[i,'pitch_type'] = 'BB'

# Encode pitch type data for correlation matrix
print('Processing one encoding...')        
#df2 = pd.get_dummies(df2, drop_first=True)
from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
df2['pitch_type'] = lab_enc.fit_transform(df2['pitch_type'].values)

# Correlation matrix <-- write to csv file
print('Writing correlation matrix to csv file...')
corr_matrix = df2.corr()
corr_matrix.to_csv("D:\OneDrive - Saginaw Valley State University\CSIS697\HW\project\corr_matrix_FB_OS.csv") #<---- YOU WILL NEED TO REWRITE NEW PATH 

# Drop additional columns
#print('Dropping additional unnecessary columns...')
df2 = df2.drop(columns=['on_3b', 'on_2b', 'on_1b',
                        'pitch_num', 'outs', 's_count',
                        'b_count', 'ab_id','b_score',
                        'event_num', 'zone','nasty', 'z0',
                        'y0','x0','x','vx0',
                        'type_confidence','sz_bot','sz_top',
                        'break_length','px'])

print('Writing to dataframe to csv file...')
df2.to_csv("D:\OneDrive - Saginaw Valley State University\CSIS697\HW\project\\fb_os_df.csv") #<---- YOU WILL NEED TO REWRITE NEW PATH 
print('Done...')

sns.set_style('whitegrid')
for col in df2.columns:
    plt.figure(figsize=(8,4.5))
    sns.scatterplot(data=df2, x=df2['pitch_type'], y=col, hue=col, palette='CMRmap', legend=False)




