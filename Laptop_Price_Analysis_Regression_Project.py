#!/usr/bin/env python
# coding: utf-8

# # LAPTOP_PRICE_ANALYSIS

# # Introduction
# 
# The Laptop Price Dataset is a comprehensive collection of data related to laptop prices, specifications, and various factors that influence the pricing of laptops. This dataset provides valuable insights into the dynamics of the laptop market, making it a valuable resource for researchers, analysts, and businesses in the technology industry.

# # Import all the liabraries -
# 
# Pandas - Used to analyse data. It has function for analysing,cleaning,exploring and manipulating data.
# 
# Numpy - Mostly work on numerical values for making Arithmatic Operations.
# 
# Matplotlib - Comprehensive library for creating static,animated and intractive visualization.
# 
# Seaborn - Seaborn is a python data visualization library based on matplotlib. It provides a high-level interface for drawing intractive and informative statastical graphics.
# 
# 
# 
# 
# 
# 
# 
# Warnings - warnings are provided to warn the developer of situation that are not necessarily exceptions and ignore them.
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# # By read_csv() function we are reading the dataset present in "laptop_train.csv" file.
# 
# Took dataset on kaggle
# 
# We will be making predictions models using Machine Learning Algorithams.

# In[2]:


df=pd.read_csv("laptops_train.csv")


# In[3]:


df


# # Performing Exploratory Data Analysis

# #### Find information about data by using df.info()
# 
# This laptops_train.csv dataset contains the information of patients health condition through their tests peroforms during covid19 with 977 rows and 13 columns.

# In[4]:


df.info()


# df.info() function display information about your DataFrame, including the data types of each column, the number of non-null values, and memory usage.
# 
# This dataset contain 977 rows and 13 columns out of there are 1 numerical column and 12 categorical column.

# #### To find how much null values in DataSet
# 
# With the help of isnull().sum() function we got the null count of all columns.

# In[5]:


df.isnull().sum()


# df.isnull() method in pandas is used to check for missing or null values in a DataFrame. 

# #### Drop Unwanted Columns

# In[6]:


df.drop("Operating System Version",axis=1,inplace=True)


# use drop method in pandas to remove a column named "Operating System Version" from your DataFrame.

# #### Change Datatype

# In[7]:


df["Screen Size"] = df["Screen Size"].replace(r'[^0-9.]', '', regex=True)
df["RAM"] = df["RAM"].replace(r'[^0-9.]', '', regex=True)
df["Weight"] = df["Weight"].replace(r'[^0-9.]', '', regex=True)


# In that removing any characters from the "Screen Size", "RAM", "Weight" column that are not digits (0-9) or a period (.). It uses a regular expression to match and replace those characters with an empty string.

# In[8]:


df["Screen Size"]=df["Screen Size"].astype("float")
df["RAM"]=df["RAM"].astype("float")
df["Weight"]=df["Weight"].astype("float")


# is converting the "Screen Size," "RAM," and "Weight" columns to the float data type.

# In[9]:


df.info()


# #### Histogram

# In[10]:


plt.figure(figsize=(10, 7))
plt.hist(df["Weight"], bins=10, edgecolor="black") 
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.title("Weight Distribution")
plt.xticks(rotation=90)
plt.show()


# #### Pieplot

# In[11]:


df["Category"].value_counts()


# In[12]:


plt.figure(figsize=(7,7))
df["Category"].value_counts().plot.pie(autopct="%1.1f%%",explode=(0,0,0,0,0,1))
plt.title("Category counts")
plt.show()


# #### Scatterplot

# In[13]:


df.corr()


# In[14]:


plt.figure(figsize=(7,7))
sns.scatterplot(data=df,x="Weight",y="Price",hue="RAM")
plt.plot()


# #### Countplot

# In[15]:


plt.figure(figsize=(10,10))
sns.countplot(data=df,x="Manufacturer")
plt.yticks(rotation=90)
plt.xticks(rotation=90)
plt.show()


# #### Find Relation between each other

# In[16]:


sns.pairplot(df)


# The pairplot function in seaborn is a great way to quickly visualize relationships between variables.

# #### Divide the Data into Numeric and Categorical form

# In[17]:


num_feature=df.select_dtypes(["int64","float64"])
cat_feature=df.select_dtypes(["object"]).columns


# Here we are saperate numerical and categorical features for perform encoding on categorical data.

# In[18]:


num_feature


# In[19]:


cat_feature


# In[20]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df[cat_feature]=oe.fit_transform(df[cat_feature])


# OrdinalEncoder is transform categorical features into numerical representations.

# In[21]:


df


# #### Splitting Data Into Features and Target

# In[22]:


x=df.iloc[:,:-1]
x


# We are splitting overall data except target column that is "target" in x variable

# In[23]:


y=df["Price"]
y


# In that separate "target" column as a target in y variable.

# #### Skewness

# In[24]:


from scipy.stats import skew


# Here we are calculate the skewness and visualize their distributions using sns.distplot from the Seaborn library. 

# In[25]:


from scipy.stats import skew

for i in x:
    print(i)
    print(skew(x[i]))
    
    plt.figure
    sns.distplot(x[i])
    plt.show()


# #### To calculate Correlation

# In[26]:


pd.concat([x,y],axis=1).corr().style.background_gradient()


# #### Apply Standard Scaler to Scale the data at one level

# In[29]:


from sklearn.preprocessing import StandardScaler


# Standardization is the process of rescaling the features so that they have the properties of a standard normal distribution with a mean of 0 and a standard deviation of 1

# In[30]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


# train_test_split function from scikit-learn to split your dataset into training and testing sets. This is a common in machine learning to evaluate the performance of a model.
# 
# In that testing data is 20% and random state is 1.

# In[31]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# We make object of standard scaler. 
# In that we have to pass xtrain and xtest for transform into standard scaler.

# In[32]:


xtrain


# This is our standardize xtrain data on which we performed Standard Scaling

# In[33]:


xtest


# This is our standardize xtest data on which we performed Standard Scaling

# #### Train_Test_Split for separating data into training and testing phase

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


# train_test_split function from scikit-learn to split your dataset into training and testing sets. This is a common in machine learning to evaluate the performance of a model.
# 
# In that testing data is 20% and random state is 1.

# #### Build a Model by using LinearRegression Algoritham

# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)


# We are using the LinearRegression class from scikit-learn to fit a linear regression model in our training data (xtrain, ytrain) and make predictions on our test data (xtest).

# #### Evaluate a model

# In[38]:


from sklearn.metrics import r2_score


# In[39]:


r2_score(ytest,ypred)


# The r2_score function compares the predicted values (ypred) with the true values (ytest) and computes the R-squared score.

# #### By applying Polynominal Linear Regression

# In[40]:


from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures()
polyx=pf.fit_transform(xtrain)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(polyx,ytrain)
ypred_train=lr.predict(polyx)


# In[41]:


from sklearn.metrics import r2_score
r2_score(ytrain,ypred_train)


# By using Polunominal Linear Regression we have 80% accuracy on training data.

# In[42]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(ytrain,ypred_train)


# In[43]:


from sklearn.metrics import mean_squared_error
mean_squared_error(ytrain,ypred_train)


# In[44]:


np.sqrt(mean_squared_error(ytrain,ypred_train))


# Here we perform lost functions on training data for reduce error rate.

# In[45]:


from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures()
polyx=pf.fit_transform(xtest)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(polyx,ytest)
ypred_test=lr.predict(polyx)


# In[46]:


from sklearn.metrics import r2_score
r2_score(ytest,ypred_test)


# By using Polunominal Linear Regression we have 73% accuracy on testing data.

# In[47]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(ytest,ypred_test)


# In[48]:


from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,ypred_test)


# In[49]:


np.sqrt(mean_squared_error(ytest,ypred_test))


# Here we perform lost functions on testing data for reduce error rate.

# #### Support Vector Regression

# In[56]:


from sklearn.svm import SVR

# Create an SVR model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Train the model
svr.fit(xtrain, ytrain)

# Make predictions
ypred_test = svr.predict(xtest)


# In[57]:


r2_score(ytest,ypred)


# Here we evaluate the performance of our SVR model using appropriate regression metrics such as R2_score.
