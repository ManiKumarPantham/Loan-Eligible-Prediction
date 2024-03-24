#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df_train = pd.read_csv(r"C:\Users\HP\Downloads\loan-train.csv")
df_train


# In[3]:


df_test = pd.read_csv(r"C:\Users\HP\Downloads\loan-test.csv")
df_test


# In[4]:


df_train.columns


# In[5]:


df_train.info()


# In[6]:


df_train.describe()


# In[7]:


df_train.head()


# In[8]:


df_train.drop(['Loan_ID'], axis = 1, inplace = True)
df_train


# In[9]:


df_test.drop(['Loan_ID'], axis = 1, inplace = True)
df_test


# In[10]:


df_train['Dependents'] = df_train['Dependents'].str.replace('3+', '3')
df_train


# In[11]:


df_test['Dependents'] = df_test['Dependents'].str.replace('3+', '3')
df_train


# In[12]:


df_train.isnull().sum()


# In[13]:


mean_impute = SimpleImputer(missing_values = np.nan, strategy = 'mean')
mode_impute = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

df_train['Gender'] = pd.DataFrame(mode_impute.fit_transform(df_train[['Gender']]))
df_train['Married'] = pd.DataFrame(mode_impute.fit_transform(df_train[['Married']]))
df_train['Self_Employed'] = pd.DataFrame(mode_impute.fit_transform(df_train[['Self_Employed']]))

df_train['Dependents'] = pd.DataFrame(mean_impute.fit_transform(df_train[['Dependents']]))
df_train['LoanAmount'] = pd.DataFrame(mean_impute.fit_transform(df_train[['LoanAmount']]))
df_train['Loan_Amount_Term'] = pd.DataFrame(mean_impute.fit_transform(df_train[['Loan_Amount_Term']]))
df_train['Credit_History'] = pd.DataFrame(mean_impute.fit_transform(df_train[['Credit_History']]))


# In[14]:


df_train.isnull().sum()


# In[15]:


df_test.isnull().sum()


# In[16]:


df_test['Gender'] = pd.DataFrame(mode_impute.fit_transform(df_test[['Gender']]))
#df_test['Married'] = pd.DataFrame(mode_impute.fit_transform(df_train[['Married']]))
df_test['Self_Employed'] = pd.DataFrame(mode_impute.fit_transform(df_test[['Self_Employed']]))

df_test['Dependents'] = pd.DataFrame(mean_impute.fit_transform(df_test[['Dependents']]))
df_test['LoanAmount'] = pd.DataFrame(mean_impute.fit_transform(df_test[['LoanAmount']]))
df_test['Loan_Amount_Term'] = pd.DataFrame(mean_impute.fit_transform(df_test[['Loan_Amount_Term']]))
df_test['Credit_History'] = pd.DataFrame(mean_impute.fit_transform(df_test[['Credit_History']]))


# In[17]:


df_test.isnull().sum()


# In[18]:


#Spliting the train data into X and y
y = df_train['Loan_Status']
X = df_train.loc[:, df_train.columns != 'Loan_Status']


# In[19]:


#Spliting the data into categorical and numerical columns
cat_columns = X.select_dtypes(include = 'object').columns
num_columns = X.select_dtypes(exclude = 'object').columns

test_cat_cols = df_test.select_dtypes(include = 'object').columns
test_num_cols = df_test.select_dtypes(exclude = 'object').columns


# In[20]:


X[cat_columns]


# In[21]:


#Converting categorical data into number format using get_dummies function
cat_modifd_col = pd.get_dummies(X[cat_columns], drop_first = True, dtype = int)
cat_modifd_col


# In[22]:


#Converting categorical data into number format using get_dummies function
df_test_modfd_cat_cols = pd.get_dummies(df_test[test_cat_cols], drop_first = True, dtype = int)
df_test_modfd_cat_cols


# In[23]:


#Concatenating the two data frames
df1 = pd.concat([X[num_columns], cat_modifd_col], axis = 1)
df1


# In[24]:


#Concatenating the two data frames
df2 = pd.concat([df_test[test_num_cols], df_test_modfd_cat_cols], axis = 1)
df2


# In[25]:


#Standardization the train data
std_scale = StandardScaler()
std_df = pd.DataFrame(std_scale.fit_transform(df1), columns = df1.columns)
std_df


# In[26]:


#Standardizing the testing data
test_std_scale = StandardScaler()
test_std_df = pd.DataFrame(test_std_scale.fit_transform(df2), columns = df2.columns)
test_std_df


# In[27]:


#Spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(std_df, y)


# In[28]:


len(X_train), len(X_test), len(y_train), len(y_test)


# In[29]:


#Building LogisticRegression algoritham
logit = LogisticRegression(random_state = 0)
logit_model = logit.fit(X_train, y_train)


# In[30]:


#prediction on test data
test_pred = logit_model.predict(X_test)
print('Test data accuracy', accuracy_score(y_test, test_pred))

#prediction on train data
train_pred = logit_model.predict(X_train)
print('Train data accuracy', accuracy_score(y_train, train_pred))


# In[31]:


#Builging KNN Clasiffier algorithm
knn_clasifier = KNeighborsClassifier(n_neighbors = 7)
knn_model = knn_clasifier.fit(X_train, y_train)   


# In[32]:


#prediction on test data
test_pred = knn_model.predict(X_test)
print('Test data accuracy', accuracy_score(y_test, test_pred))

#prediction on train data
train_pred = knn_model.predict(X_train)
print('Train data accuracy', accuracy_score(y_train, train_pred))


# In[33]:


#Building Decision Tress Algorithm
dtree_clasifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 10)
dtree_model = dtree_clasifier.fit(X_train, y_train)


# In[34]:


#prediction on test data
test_pred = dtree_model.predict(X_test)
print('Test data accuracy', accuracy_score(y_test, test_pred))

#prediction on train data
train_pred = dtree_model.predict(X_train)
print('Train data accuracy', accuracy_score(y_train, train_pred))


# ##### Among KNN, Logistic Regression and Decession Tree algorithams, KNN gives the best results. However, we are considering KNN is the best model and using it for actual test data predictions. 

# In[35]:


#Prediction on ACTUAL TEST Data
final_test_pred = knn_model.predict(test_std_df)
print('Acutal Test data predictions', final_test_pred)


# In[36]:


#Crerating a new columns to Acutal Test Data Dataset and storing predicted values
df_test['Loan_status'] = pd.Series(final_test_pred)
df_test

