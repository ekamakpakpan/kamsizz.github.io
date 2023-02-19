#!/usr/bin/env python
# coding: utf-8

# In[38]:


# data analysis libraries
import numpy as np
import pandas as pd

# visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pwd()


# In[3]:


#  importing dataset
data = pd.read_csv(r'C:\Users\Favour Ekam\Documents\data science learning resource\CAPSTONE PROJECT Online Payment Fraud Detection.csv')
data.head()


# In[4]:


data.tail()


# FEATURES OF DATABASE AND MEANINGS
# 
# -step: represents a unit of time where 1 step equals 1 hour
# 
# -type: type of online transaction
# 
# -amount: the amount of the transaction
# 
# -nameOrig: customer starting the transaction
# 
# -oldbalanceOrg: balance before the transaction
# 
# -newbalanceOrig: balance after the transaction
# 
# -nameDest: recipient of the transaction
# 
# -oldbalanceDest: initial balance of recipient before the transaction
# 
# -newbalanceDest: the new balance of the recipient after the transaction
# 
# -isFraud: fraud transaction
# 
# Type Markdown and LaTeX: 𝛼2

# In[5]:


# getting the shape of data(rows and columns)
data.shape


# In[6]:


# info on our data

data.info()


# In[7]:


# descriptive statistics- this gives descriptive stats of only numerical var.
data.describe()


# In[8]:


# checking for missing values

data.isnull().sum()


# In[9]:


# confirming the data type

type(data)


# In[10]:


# checking for duplicates

data.duplicated().sum()


# In[11]:


type_counts=data.type.value_counts()
type_counts


# In[12]:


nameOrig_counts=data.nameOrig.value_counts()
nameOrig_counts


# In[13]:


isFraud_counts=data.isFraud.value_counts()
isFraud_counts


# In[14]:


# univariate-exploring just one feature
plt.figure(figsize=(8,6))
plt.title('Types of online transactions')
sns.countplot(data=data, x='type')
plt.xlabel('type')
plt.ylabel('Count of transaction types')
plt.show()


# In[ ]:





# In[15]:


# bivariate analysis

plt.figure(figsize=(8,6))
plt.title('Amount based on transaction')
sns.barplot(x='type',y='amount', data=data);
plt.xlabel('amount')
plt.ylabel('type')
plt.show()


# In[16]:


plt.figure(figsize=(8,6))
plt.title('Amount based on transaction and isFraud')
sns.barplot(x='type',y='amount', hue='isFraud', data=data);
plt.xlabel('type')
plt.ylabel('amount')
plt.show()


# In[17]:


plt.figure(figsize=(8,6))
plt.title('Isfraud based on transaction type')
sns.barplot(x='type',y='isFraud', data=data);
plt.xlabel('type')
plt.ylabel('isFraud')
plt.show()


# In[18]:


plt.figure(figsize=(8,6))
plt.title('relationship between new and old balance')
sns.scatterplot(x='newbalanceOrig', y='oldbalanceOrg', data=data);
plt.xlabel('newbalanceOrig')
plt.ylabel('oldbalanceOrg')
plt.show()


# In[19]:


plt.figure(figsize=(8,6))
plt.title('relationship between new and old balance')
sns.scatterplot(x='newbalanceDest', y='oldbalanceDest', data=data);
plt.xlabel('newbalanceDest')
plt.ylabel('oldbalanceDest')
plt.show()


# In[20]:


sns.pairplot(data)


# ## Data Preprocessing

# In[21]:


#variable encoding
data.columns


# In[22]:


#data segmentation and droping of irrelevant features
target = data['isFraud']
data = data.drop(columns=['isFraud','step', 'nameOrig','nameDest'],axis=1)


# In[23]:


#variable encoding
data.columns


# In[24]:


#one hot encoding is used to represent nominal data type and not ordinal 8ewuy
data = pd.get_dummies(data)
data


# ## Lets scale using standardscaler

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)


# ## Training and Testing Data
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "isFraud" column.
# 

# In[26]:


# Using model_selection.train_test_split from sklearn to split the data into training and testing sets. 

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)


# ## Training the Model- using Linear regression
# Now its time to train our model on our training data!
# 

# In[27]:


#Import LinearRegression from sklearn.linear_model 

from sklearn.linear_model import LinearRegression


# In[28]:


# Create an instance of a LinearRegression() model named lin_mod.

model = LinearRegression()


# In[29]:


#Train/fit lin_mod on the training data.

model.fit(X_train,y_train)


# In[30]:


# Print out the coefficients of the model

print(model.coef_)
print(model.intercept_)


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!

# In[31]:


# Use lin_mod.predict() to predict off the X_test set of the data.

predictions = model.predict(x_test)


# In[32]:


#  Create a scatterplot of the real test values versus the predicted values

plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')


# ## Evaluating the Model
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# * Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error

# In[33]:


from sklearn import metrics


# In[34]:


import numpy as np

mae = metrics.mean_absolute_error(y_test,predictions)
mse = metrics.mean_squared_error(y_test,predictions)
r2_score = metrics.r2_score(y_test,predictions)
rmse = np.sqrt(mse)


# In[35]:


print(r2_score)
print(mse)
print(mae)
print(rmse)


# ## Training the Model- using Logistic regression

# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


clf = LogisticRegression()
clf.fit(X_train,y_train)


# In[55]:


prediction = clf.predict(x_test)
prediction


# In[47]:


from sklearn.metrics import roc_auc_score


# In[48]:


score = roc_auc_score(y_test,prediction)


# In[49]:


score


# In[50]:


#naive bayes model
from sklearn.naive_bayes import GaussianNB


# In[51]:


nb = GaussianNB()
nb.fit(X_train,y_train)


# In[60]:


nb_pred = nb.predict(x_test)
nb_pred


# In[53]:


score2 = roc_auc_score(y_test,nb_pred)


# In[54]:


score


# In[58]:


#confusion metric-

from sklearn.metrics import confusion_matrix


# In[64]:


confusion_matrix = metrics.confusion_matrix(y_test, prediction)


# In[65]:


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])


# In[66]:


cm_display.plot()
plt.show()


# In[72]:


# calculating Accuracy-Accuracy measures how often the model is correct.

Accuracy = metrics.accuracy_score(y_test, prediction)
Accuracy


# In[75]:


# Precision- Of the positives predicted, what percentage is truly positive

Precision = metrics.precision_score(y_test, prediction)
Precision


# In[78]:


# Sensitivity (Recall)- Of all the positive cases, what percentage are predicted positive? Sensitivity (sometimes called Recall) 
# measures how good the model is at predicting positives.

Sensitivity_recall = metrics.recall_score(y_test, prediction)
Sensitivity_recall


# In[79]:


# Specificity- How well the model is at prediciting negative results?
# Specificity is similar to sensitivity, but looks at it from the persepctive of negative results.

Specificity = metrics.recall_score(y_test, prediction, pos_label=0)
Specificity


# In[ ]:


# In conclusion, the Confusion model is the best in predicting Fraud becauses it has an accuracy of 99%.
# The business should be more concerned with better results from false negatives than True positives.

