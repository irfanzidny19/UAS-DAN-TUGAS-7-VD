#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NAMA : IRFAN ZIDNY
# NIM  : 41819010073


# In[2]:


import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd

# Read the given CSV file, and view some sample records
advertising = pd.read_csv("Company_data.csv")
advertising


# In[3]:


advertising.shape

# Info our dataset
advertising.info()

# Describe our dataset
advertising.describe()


# In[4]:


import matplotlib.pyplot as plt 
import seaborn as sns

# Using pairplot we'll visualize the data for correlation
sns.pairplot(advertising, x_vars=['TV', 'Radio','Newspaper'], 
             y_vars='Sales', size=4, aspect=1, kind='scatter')
plt.show()


# In[5]:


sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[7]:


pip install sklearn


# In[8]:


X = advertising['TV']
y = advertising['Sales']


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, 
                                                    test_size = 0.3, random_state = 100)


# In[10]:


X_train
y_train


# In[11]:


pip install statsmodels


# In[12]:


import statsmodels.api as sm

# Adding a constant to get an intercept
X_train_sm = sm.add_constant(X_train)


# In[13]:


lr = sm.OLS(y_train, X_train_sm).fit()

# Printing the parameters
lr.params


# In[14]:


lr.summary()


# In[15]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[16]:


y_train_pred = lr.predict(X_train_sm)

# Creating residuals from the y_train data and predicted y_data
res = (y_train - y_train_pred)


# In[17]:


fig = plt.figure()
sns.distplot(res, bins = 15)
plt.title('Error Terms', fontsize = 15)
plt.xlabel('y_train - y_train_pred', fontsize = 15)
plt.show()


# In[18]:


plt.scatter(X_train,res)
plt.show()


# In[19]:


X_test_sm = sm.add_constant(X_test)

# Predicting the y values corresponding to X_test_sm
y_test_pred = lr.predict(X_test_sm)

# Printing the first 15 predicted values
y_test_pred


# In[20]:


from sklearn.metrics import r2_score

# Checking the R-squared value
r_squared = r2_score(y_test, y_test_pred)
r_squared


# In[21]:


plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()


# In[30]:


from sklearn.metrics import r2_score

# Checking the R-squared value
r_squared = r2_score(y_test, y_test_pred)
r_squared


# In[35]:



X_train_lm.shape


X_train_lm = X_train_lm.values.reshape(-1,1)
X_test_lm = X_test_lm.values.reshape(-1,1)

print(X_train_lm.shape)
print(X_test_lm.shape)


# In[34]:


from sklearn.model_selection import train_test_split
X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y,train_size = 0.7, test_size = 0.3, random_state = 100)


# In[36]:


from sklearn.linear_model import LinearRegression

# Creating an object of Linear Regression
lm = LinearRegression()

# Fit the model using .fit() method
lm.fit(X_train_lm, y_train_lm)


# In[37]:


print("Intercept :",lm.intercept_)

# Slope value
print('Slope :',lm.coef_)


# In[38]:


y_train_pred = lm.predict(X_train_lm)
y_test_pred = lm.predict(X_test_lm)

# Comparing the r2 value of both train and test data
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))


# In[39]:


# LINK GITHUB : https://github.com/irfanzidny19


# In[ ]:




