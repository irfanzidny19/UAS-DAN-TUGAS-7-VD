#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NAMA : IRFAN ZIDNY
# NIM  : 41819010073


# In[36]:


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[6]:


pip install sklearn


# In[7]:


dataFrame = pd.read_csv("testcpns.csv")
dataFrame.describe()


# In[13]:


import matplotlib.pyplot as plt 
import seaborn as sns

# Using pairplot we'll visualize the data for correlation
sns.pairplot(dataFrame, x_vars=['ipk'], 
             y_vars='toefl', height=4, aspect=1, kind='scatter', hue='diterima')
plt.show()


# In[14]:


sns.pairplot(dataFrame, x_vars=['pengalaman_kerja'], 
             y_vars='toefl', height=4, aspect=1, kind='scatter', hue='diterima')
plt.show()


# In[15]:


sns.pairplot(dataFrame, x_vars=['pengalaman_kerja'], 
             y_vars='ipk', height=4, aspect=1, kind='scatter', hue='diterima')
plt.show()


# In[16]:


sns.pairplot(dataFrame, x_vars=['ipk', 'pengalaman_kerja','toefl'], 
             y_vars='diterima', height=4, aspect=1, kind='scatter')
plt.show()


# In[30]:


X = dataFrame.iloc[:, :-1]
y = dataFrame.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[32]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[33]:


y_pred = model.predict(X_test)


# In[34]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[37]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[24]:


dataFrame = pd.read_csv("testcpns.csv")
dataFrame


# In[38]:


# LINK GITHUB : https://github.com/irfanzidny19


# In[ ]:




