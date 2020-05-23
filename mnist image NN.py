#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


train , test = dataset


# In[4]:


X_train , y_train = train


# In[5]:


X_test , y_test = test


# In[6]:


X_train.shape


# In[7]:


img = X_train[8]


# In[8]:


img.shape


# In[9]:


y_train[8]


# In[10]:


import matplotlib.pyplot as py


# In[11]:


py.imshow(img)


# In[12]:


X_train_1d = X_train.reshape(-1 , 28*28)


# In[13]:


X_test_1d = X_test.reshape(-1 , 28*28)


# In[14]:


X_train_1d.shape


# In[15]:


X_train = X_train_1d.astype('float32')


# In[16]:


X_test = X_test_1d.astype('float32')


# In[17]:


#X_test.shape


# In[18]:


from keras.utils.np_utils import to_categorical


# In[19]:


y_train = to_categorical(y_train)


# In[20]:


y_test = to_categorical(y_test)


# In[21]:





# In[ ]:





# In[22]:


from keras.models import Sequential


# In[23]:


model = Sequential()


# In[24]:


from keras.layers import Dense


# In[25]:


model.add(Dense(units = 210 , input_dim = 28*28 , activation='relu'))


# In[26]:


model.summary()


# In[27]:


model.add(Dense(units = 120 , activation='relu'))


# In[28]:


model.summary()


# In[29]:


model.add(Dense(units = 100 ,  activation='relu'))


# In[30]:


model.add(Dense(units = 10 , activation='softmax'))


# In[31]:


from keras.optimizers import Adam


# In[32]:


model.compile(optimizer=Adam(learning_rate = 0.001) , loss = 'categorical_crossentropy',
             metrics = ['accuracy'])


# In[33]:


h = model.fit(X_train , y_train , epochs = 20)


# In[ ]:


#img1 = X_test[12]


# In[ ]:


#y_test[12]


# In[ ]:


#py.imshow(img1)


# In[ ]:


#model.predict(img1)


# In[34]:


y_pred = model.predict(X_test)


# In[ ]:


#from sklearn.metrics import confusion_matrix


# In[35]:


y_pred[0] , y_train[0]


# In[36]:


X_test[0].reshape(784,1)


# In[37]:


X_train[0].reshape(784,1)


# In[43]:


py.imshow(X_test[15].reshape(28,28))


# In[40]:


y_test[15]


# In[ ]:




