#!/usr/bin/env python
# coding: utf-8

# # ML Project

# ## Load Data

# In[5]:


import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
df


# ## Data Preperation

# ### Data seperation as X and Y

# In[7]:


y = df['logS']
y


# In[8]:


X = df.drop( 'logS' , axis = 1)
X


# ### Data Splitting

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=100)


# In[14]:


X_train


# In[15]:


X_test


# ## Model Building

# ### Linear Regression

# In[16]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[17]:


y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)


# In[18]:


y_lr_train_pred


# In[19]:


y_lr_test_pred


# In[20]:


from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)


# In[21]:


lr_train_mse


# In[22]:


lr_train_r2


# In[23]:


lr_test_mse


# In[24]:


lr_test_r2


# In[27]:


lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


# In[28]:


lr_results


# ### Random Forest

# In[29]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)


# In[38]:


y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# In[39]:


from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


# In[40]:


rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
rf_results


# ## Model Comparison

# In[41]:


df_models = pd.concat([lr_results, rf_results], axis=0)
df_models


# In[42]:


df_models.reset_index(drop=True)


# # Data visualization of prediction results

# In[48]:


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c='#7CAE00', alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')


# In[ ]:




