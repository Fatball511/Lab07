#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:11:05 2022

@author: keithcheng
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import warnings 
warnings.simplefilter("ignore")


# 1. Load the data in a DataFrame
df = pd.read_csv('censusCrimeClean.csv')
# 2. extract all the data except the first column, save it as a variable 'Community'
Community = df.iloc[:,0]
df = df.iloc[:,1:]

# 3. Create and fit PCA with two components with these features
pca = PCA(n_components=2)
pca.fit(df)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.components_)
# 4. Try standardizing the variables
sklearn.preprocessing.scale(df,axis=1)#standardize the data

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


scaled_features = StandardScaler().fit_transform(df.values)

scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


scaled_features_df['communityname'] = Community
scaled_features_df = scaled_features_df.T
scaled_features_df = scaled_features_df.abs()
print(scaled_features_df)