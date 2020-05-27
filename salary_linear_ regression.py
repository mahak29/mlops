#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


dataset = pd.read_csv('SalaryData.csv')


# In[32]:


dataset.head()


# In[33]:


dataset.info()


# In[ ]:





# In[4]:


y = dataset['Salary']


# In[5]:


x = dataset['YearsExperience']


# In[6]:


type(x)


# In[7]:


X = x.values.reshape(30,1)


# In[34]:


X.shape


# In[35]:


y.shape


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


# shift + tab


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


model = LinearRegression()

# y = b + c x


# In[41]:


#  know : x , y 

# find: c and b

# MAE : error : min error is the best
model.fit(X_train, y_train)


# In[46]:


y_pred = model.predict(X_test)


# In[47]:


y_pred


# In[48]:


y_test


# In[53]:


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')


# In[14]:


model.coef_


# In[15]:


model.intercept_


# In[16]:


# linear regression : linear algebra : linear function
# y = b + cx
# y = b + 9449 x


# In[17]:


1.1*9449.96232146 + b


# In[ ]:


1.5*9449.96232146


# In[ ]:


# fresh , exp=0
# exp = x
# y = b + cx
# weight = c = coefficient

# fresh: initial salary offer == constant = b = bias
# y = b

# weight * 1.1


y= 25792 +  9449 * 1.1


# In[ ]:





# In[23]:


from sklearn import metrics


# In[55]:


# close to zero, was best model : ideal case
# MAE

# loss function / error 
metrics.mean_absolute_error(y_test,y_pred)


# In[56]:


# MSE : penalty

# better to use
metrics.mean_squared_error(y_test ,y_pred)


# In[ ]:


# RMSE


# In[ ]:


from sklearn.externals import joblib


# In[ ]:


joblib.dump(model, 'salary_model.pk1')


# In[ ]:





# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[20]:


plt.scatter(X,y)


# In[ ]:




