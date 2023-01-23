#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy import stats
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta
from tensorflow.keras.layers import LeakyReLU, PReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import missingno as msno


# In[5]:


raw_data = pd.read_csv('/kaggle/input/new-loan-tin100/SBAnational.csv', delimiter=',', low_memory=False)


# In[6]:


raw_data.head()


# In[8]:


raw_data.describe(include='object').T


# In[9]:


msno.matrix(raw_data, labels=True, sort='descending')


# In[19]:


def wrangle(df):
  # copy
  cleaned_df = df.copy(deep = True)
 
  ## Remove symbols
  list_s = ['DisbursementGross','GrAppv','SBA_Appv']

  for colm in list_s:
    cleaned_df[colm] = cleaned_df[colm].str.replace('[$,]', '').astype(float)
   
       
  ## List of unnecessary, inconsistent and leaky variables
  list_un = ['LoanNr_ChkDgt','Name','City','State', 'Zip','Bank',
              'BankState','NAICS','ApprovalDate','NewExist', 'DisbursementDate',
              'ApprovalFY', 'FranchiseCode','ChgOffPrinGr','ChgOffDate',
              'BalanceGross','RevLineCr','LowDoc','CreateJob','RetainedJob','UrbanRural'
              ]
    
  # Create a Real State variable 
  list_real = []
    
  for real in cleaned_df['Term']:
    if real >=240:
      real = 1
    else:
      real = 0

    list_real.append(real)
        
  cleaned_df['RealEstate'] = list_real
   


  # SBA's Guaranteed Portion of Approved Loan
  cleaned_df['portion'] = cleaned_df['SBA_Appv']/ cleaned_df['GrAppv']
   
   
  # New target variable
  cleaned_df['Default'] = (cleaned_df['MIS_Status'] == 'CHGOFF').astype(int)
   
  #Remove old target variable
  list_un.append('MIS_Status')
   
  # Drop columns
  cleaned_df.drop(columns = list_un, inplace = True)

  # drop miss_values
  cleaned_df.dropna(inplace = True)
     
  return cleaned_df


# In[20]:


df_tran = wrangle(raw_data)
print(df_tran.shape)


# In[21]:


df_tran.head(10)


# In[22]:


df_tran.describe().round(2)


# ### Splitting the data into train and test

# In[23]:


training_data = df_tran.sample(frac=0.8, random_state=21)
testing_data = df_tran.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[24]:


# Code for scikit-learn based model

X = training_data.drop(["Default"], axis=1)
y = training_data["Default"]


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.3, random_state=21)

print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# In[26]:


# Scaling the data with StandardScaler()

sc = StandardScaler()
sc.fit(X_train)


# Transform (standardise) both X_train and X_test with mean and STD from 
# training data

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)


# ### ANN with Keras

# In[27]:


# Code for creating and training a ANN with Keras

model = models.Sequential([
    layers.Dense(1024, activation ="PReLU"),
    layers.Dense(512, activation ="PReLU"),
    layers.Dropout(0.1),
    layers.Dense(256, activation ="PReLU"),
    layers.Dropout(0.1),
    layers.Dense(128, activation = "PReLU"),
    layers.Dense(64, activation = "PReLU"),
    layers.Dense(1, activation = "sigmoid")])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = EarlyStopping(monitor='loss', patience=5)

# Fit model (in the same manner as you would with scikit-learn)
model.fit(X_train_sc, 
          y_train,
          epochs=100,
          callbacks=callback,
          batch_size=256,
          validation_split=0.3)


# In[28]:


model.summary()


# ### Saving our best model to not train it every time

# In[29]:


model.save("tin100-ann-model.h5") # Saving to a h5 file

