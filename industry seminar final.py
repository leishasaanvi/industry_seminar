#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[49]:


data=pd.read_csv("industryseminar1(1).csv")


# In[50]:


data.head()


# In[51]:


print(data.isnull().sum())


# In[52]:


data.tail()


# In[78]:


#Outlier Detection
# Example: Box plot for outlier detection (replace 'C_api' with your feature)
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, y='gender')
plt.title('Box Plot for Outlier Detection (gender)')
plt.ylabel('gender')
plt.show()


# In[79]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=data, y='C_api')
plt.title('Box Plot for Outlier Detection (C_api)')
plt.ylabel('C_api')
plt.show()


# In[54]:


# Data Summary
print(data.head())  # Display the first few rows
print(data.info())  # Check data types and missing values
print(data.describe())  # Summary statistics for numerical columns


# In[55]:


#Class Distribution
class_distribution = data['gender'].value_counts()
print(class_distribution)


# In[56]:


#gender analysis
sns.countplot(data=data, x='gender')
plt.title('Gender Distribution')
plt.show()


# In[57]:


#Feature Analysis
# Example: Box plot of a numerical feature by gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='gender', y='C_api')
plt.title('Box Plot of C_api by Gender')
plt.xlabel('Gender')
plt.ylabel('C_api')
plt.show()


# In[58]:


# Example: Count plot of a categorical feature by gender
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='E_Bpag', hue='gender')
plt.title('Count Plot of E_Bpag by Gender')
plt.xlabel('E_Bpag')
plt.ylabel('Count')
plt.show()


# In[59]:


# Data Visualization
# Example: Histogram of numerical feature
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='C_api', hue='gender', kde=True)
plt.title('Distribution of C_api by Gender')
plt.xlabel('C_api')
plt.ylabel('Count')
plt.show()


# In[60]:


# Example: Bar chart of a categorical feature
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='E_NEds', hue='gender')
plt.title('Distribution of E_NEds by Gender')
plt.xlabel('E_NEds')
plt.ylabel('Count')
plt.show()


# In[61]:


# Summary statistics by gender
gender_summary = data.groupby('gender').describe()
print(gender_summary)


# In[62]:


# Example: Gender gap in the number of pages created
sns.boxplot(data=data, x='gender', y='NPcreated')
plt.title('Gender Gap in Pages Created')
plt.show()


# In[63]:


data=data.drop(['C_api','C_man'],axis = 1)
data.head()


# In[64]:


x = data.drop("gender",axis=1)


# In[65]:


y=data["gender"]


# In[66]:


scaler_ = StandardScaler()
X = pd.DataFrame(scaler_.fit_transform(x),columns = x.columns)


# In[67]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[68]:


print(X.shape,X_train.shape,X_test.shape)


# In[69]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train,y_train)


# In[70]:


y_pred = reg.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test,y_pred)*100)


# In[72]:


# Import the necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppose you have a dataset with features X and labels y
# X represents the features, and y represents the labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}".format(accuracy))


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[74]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)


# In[75]:


y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}".format(accuracy))


# In[76]:


# Import the necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppose you have a dataset with features X and binary labels y
# X represents the features, and y represents the labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}".format(accuracy))


# In[77]:


# Import the necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppose you have a dataset with features X and binary labels y
# X represents the features, and y represents the labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}".format(accuracy))


# In[ ]:




