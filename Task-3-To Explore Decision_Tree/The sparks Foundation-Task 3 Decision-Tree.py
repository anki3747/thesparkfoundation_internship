#!/usr/bin/env python
# coding: utf-8

# # Submitted by Ankit Singh

# # Decision Tree
# 
# ### TASK 3 
# **To Explore Decision Tree Algorithm** 
# 
# For the given 'Iris' dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# This notebooks deals with understanding the working of decision trees.

# ## Importing the Required Libraries and Reading the data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import plot_tree

# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import warnings

warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('G:\project(final year)\\Iris.csv', index_col = 0)
df.head()


# In[3]:


#looking for imbalance in the dataset
df.info()


# In[4]:


target = df['Species']
df1 = df.copy()
df1 = df1.drop('Species', axis =1)
df1.shape


# In[5]:


# No Null values observed 
# let's plot pair plot to visualise the attributes all at once

sns.pairplot(df, hue = 'Species')


# We can easily observe that "iris-setosa" makes a distinctive cluster in every parameter, while the other two species are overlapping a bit on each other.

# In[6]:


# correlation matrix
sns.heatmap(df.corr())


# Observations made - 
# 1. Petal length is highly related to petal width.
# 2. Sepal lenth is not related sepal width.

# In[7]:


# Defining the attributes and labels

X = df.iloc[:, [0, 1, 2, 3]].values #Attributes

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

y = df['Species'].values  #Labels

print("The shape of the data is-", df.shape)


# ## Model Training

# Let us split the data into test and train for trainig our model.

# In[8]:


# Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42 )

print("Training split - ", X_train.shape)
print("Testing split - ", X_test.shape)


# In[9]:


# Defining the decision tree algorithm

dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

print('Decision Tree Classifer Created')


# In[10]:


# Predicting the values of test data
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))


# In[11]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');

plt.ylabel('Actual label');
plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)


# In[ ]:





# In[12]:


# Visualising the graph without the use of graphviz

plt.figure(figsize = (20,20))
dec_tree = plot_tree(dtree, feature_names = df1.columns, 
                     class_names = target.values, filled = True , precision = 4, rounded = True);


# This concludes this notebook.

# 

# In[ ]:




