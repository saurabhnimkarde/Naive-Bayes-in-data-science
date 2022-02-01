#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


salary=pd.read_csv('SalaryData_Train.csv')


# In[3]:


salary.head()


# In[4]:


salary.info()


# In[8]:


salary['workclass']=salary['workclass'].astype('category')
salary['education']=salary['education'].astype('category')
salary['maritalstatus']=salary['maritalstatus'].astype('category')
salary['occupation']=salary['occupation'].astype('category')
salary['relationship']=salary['relationship'].astype('category')
salary['race']=salary['race'].astype('category')
salary['native']=salary['native'].astype('category')
salary['sex']=salary['sex'].astype('category')


# In[5]:


salary.dtypes


# In[6]:


from sklearn import preprocessing                      
label_encoder = preprocessing.LabelEncoder()


# we need Salary string type data into binary numbers

# In[7]:


salary['Salary'] = label_encoder.fit_transform(salary['Salary'])


# In[8]:


salary.Salary


# we also need to convert categories into numbers

# In[9]:


salary['workclass'] = label_encoder.fit_transform(salary['workclass'])
salary['education'] = label_encoder.fit_transform(salary['education'])
salary['maritalstatus'] = label_encoder.fit_transform(salary['maritalstatus'])
salary['occupation'] = label_encoder.fit_transform(salary['occupation'])
salary['relationship'] = label_encoder.fit_transform(salary['relationship'])
salary['race'] = label_encoder.fit_transform(salary['race'])
salary['sex'] = label_encoder.fit_transform(salary['sex'])
salary['native'] = label_encoder.fit_transform(salary['native'])


# In[10]:


salary


# In[11]:


# Splitting the data into x and y as input and output

X = salary.iloc[:,0:13]
Y = salary.iloc[:,13]


# In[12]:


X


# In[13]:


Y


# In[14]:


salary.Salary.unique()


# In[15]:


salary.Salary.value_counts()


# In[16]:


# Splitting the data into training and test dataset

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# In[17]:


gnb=GaussianNB()


# In[18]:


gnb.fit(x_train,y_train)


# In[19]:


y_pred = gnb.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)


# In[20]:


y_pred


# In[21]:


pd.crosstab(y_pred,y_test)


# In[ ]:





# In[ ]:




