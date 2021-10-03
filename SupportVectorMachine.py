#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
import collections
from sklearn.svm import SVC


# In[2]:


trendingNonTrendingVideos = pd.read_csv ('C:/Users/pc/Desktop/Youtube/TrendingNonTrendingVideos.csv',header=None);
trendingNonTrendingVideos.head()


# In[3]:


trendingNonTrendingVideos = trendingNonTrendingVideos.drop(trendingNonTrendingVideos.columns[0], axis=1)
trendingNonTrendingVideos = trendingNonTrendingVideos.drop(0)
trendingNonTrendingVideos.head()


# In[4]:


trendingNonTrendingVideos[1] = pd.to_numeric(trendingNonTrendingVideos[1])
trendingNonTrendingVideos[2] = pd.to_numeric(trendingNonTrendingVideos[2])
trendingNonTrendingVideos[3] = pd.to_numeric(trendingNonTrendingVideos[3])
trendingNonTrendingVideos[4] = pd.to_numeric(trendingNonTrendingVideos[4])
trendingNonTrendingVideos[5] = pd.to_numeric(trendingNonTrendingVideos[5])
trendingNonTrendingVideos[6] = pd.to_numeric(trendingNonTrendingVideos[6])


print(trendingNonTrendingVideos.info())


# In[5]:


feature_cols = [1,2,3,4,5]
X = trendingNonTrendingVideos[feature_cols] # Features
y = trendingNonTrendingVideos[6] # Target variable


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=5, stratify=trendingNonTrendingVideos[6]) 


# In[ ]:



svclassifier = SVC(gamma='auto')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(classification_report(y_test,y_pred))


# In[99]:


smt = SMOTE()
X_train_sm,y_train_sm = smt.fit_resample(X_train,y_train)

svclassifier = SVC(gamma='auto')
svclassifier.fit(X_train_sm, y_train_sm)

y_pred = svclassifier.predict(X_test)


# In[8]:


svclassifier = SVC(gamma=0.001, C=100)
svclassifier.fit(X_train_sm, y_train_sm)

y_pred = svclassifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# In[9]:


y_pred = svclassifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(classification_report(y_test,y_pred))


# In[ ]:




