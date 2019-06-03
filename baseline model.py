# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:10:47 2019

@author: lisha
"""
import tensorflow as tf
import pandas as pd
import numpy as np


# In[2]:


X_train = pd.read_csv("x_train.csv",index_col=0)
X_test = pd.read_csv("x_test.csv",index_col=0)
y_test = pd.read_csv("y_test.csv",index_col=0)
y_train = pd.read_csv("y_train.csv",index_col=0)

from sklearn.ensemble import RandomForestClassifier


# In[539]:


clf = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=0)
clf.fit(X_train, y_train)


# In[540]:


predicted = clf.predict(X_test)


# In[541]:


set(predicted)


# In[542]:


sum(predicted == y_test)


# In[543]:


len(predicted)


# In[544]:


accuracy = 5709/10382
accuracy


# In[545]:


predicted = predicted.astype(int)
y_test = y_test.astype(int)
from sklearn.metrics import confusion_matrix
y_true = y_test
y_pred = predicted
confusion_matrix(y_true, y_pred, labels=[-1, 1, 0])
