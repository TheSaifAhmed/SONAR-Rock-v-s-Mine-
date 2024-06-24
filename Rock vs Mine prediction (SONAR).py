#!/usr/bin/env python
# coding: utf-8

# # SONAR ROCK V/S MINE PREDICTION

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #will help in splitting the data into trainanad test 
from sklearn.linear_model import LogisticRegression #calling logistic regression fro sklearn
from sklearn.metrics import accuracy_score #accuracy score helps in telling the accuracy of the model


# # Data collection and processing

# In[2]:


sonar_data= pd.read_csv("C:\\Users\\saifa\\Downloads\\Copy of sonar data.csv", header=None)


# In[3]:


sonar_data.head()


# The first 60 columns tells features of the rock and mine(denoted by R & M in 61 clomun)  

# In[4]:


sonar_data.shape


# 208 is data, and out of 61, one column is of R& M and rest 60 columns are features/intances of the data and 208 are the intances/examples of the data  

# In[5]:


sonar_data.describe() #gives the statistical measure


# In[6]:


sonar_data[60].value_counts() #it will count the no. of times R & M appears in the data


# If there had been a big difference in the counts of rock and mines our prediction would have been bad.

# In[7]:


sonar_data.groupby(60).mean()


# we clearly see that the mean of Mines are greater than that of rocks.
# 

# # Sperating Data and Labels

# this is supervised machine learning model and in such models we need to separate labels and data.
# 
# Labels are R(rock) & M(mine).
# 
# Note: In Unspervised Learning we don't need to separate data and labels 

# In[8]:


X=sonar_data.drop(columns=60, axis=1) #Agar row drop karenge toh axis=0 likhte hai
Y=sonar_data[60]
#x is data of features yaa simply data
#y is the data of labels 


# In[9]:


print(X)
print(Y)


# In[10]:


type(X)


# # Spliting Data inton Train and Test data 

# In[11]:


X_train,X_test, Y_train,Y_test =train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)
#jo variables(x_train,x_test etc) diye hai unka order matter krta hai. 
#test_size implies the amount or percentage of data you want to convert to test data; it lies b/w 0.0 to 1.0; it takes boolean value. 0.1 implies that 10% of the data is converted into Test data and 90% data will be train data 
#stratify implies with respect which label we want tro split the data.


# In[12]:


print(X.shape, X_train.shape, X_test.shape, )


# In[13]:


print(X_train, Y_train)


# # training Logistic Regression model with the training data

# In[14]:


model= LogisticRegression()


# In[15]:


model.fit(X_train, Y_train)


# # Model Accuracy

# acuracy on training data

# In[16]:


X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[17]:


print(training_data_accuracy)


# Accuracy on test data

# In[18]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)


# In[19]:


print(test_data_accuracy)


# # Model Prediction

# We have trained our model using training and tested our model with test data; now we are willing to have predictive system to predict an object is rock or mine using Sonar Data 

# In[35]:


input_data=(0.0286,0.0453,0.0277,0.0174,0.0384,0.099,0.1201,0.1833,0.2105,0.3039,0.2988,0.425,0.6343,0.8198,1,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.589,0.2872,0.2043,0.5782,0.5389,0.375,0.3411,0.5067,0.558,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.265,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062
) 
#Now we will change our data to a numpy array, as it is quite faster as compared to list/tuple and input_data is list/tuple
input_data_as_numpy_array=np.asarray(input_data)

#reshape the np array as we are predicting only for one instance or feature
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print("Rock")
else:
    print("Mine")
    


# In[29]:


r=[1,5,6,9,5,2,52,505]
r=np.asarray(r)
x=r.reshape(1,-1)
print(x,r)


# In[30]:


n=(0.0286,0.0453,0.0277,0.0174	0.0384	0.099	0.1201	0.1833	0.2105	0.3039	0.2988	0.425	0.6343	0.8198	1	0.9988	0.9508	0.9025	0.7234	0.5122	0.2074	0.3985	0.589	0.2872	0.2043	0.5782	0.5389	0.375	0.3411	0.5067	0.558	0.4778	0.3299	0.2198	0.1407	0.2856	0.3807	0.4158	0.4054	0.3296	0.2707	0.265	0.0723	0.1238	0.1192	0.1089	0.0623	0.0494	0.0264	0.0081	0.0104	0.0045	0.0014	0.0038	0.0013	0.0089	0.0057	0.0027	0.0051	0.0062	R
)


# In[31]:


type(n)


# In[ ]:




