#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


# read datset
data = pd.read_csv('Application_Data.csv')


# In[18]:


data.head(20)


# In[21]:


pd.set_option('display.max_rows', None, 'display.max_columns', None)


# In[22]:


# data preview and information
print(data.head(20))
# data.info()
# data.describe()


# In[23]:


# check unique values for each variable

# for col in data:
#     print(col)
#     print(data[col].nunique())


# In[24]:


# check unique values for each variable

for col in data:
    print(col)
    print(data[col].unique())


# In[25]:


data.columns


# In[26]:


# check unique values and thier count for each variable

# for col in col:
#     values, count = np.unique(data[col], return_counts = True)
#     print(data[col].value_counts())


# In[27]:


# drop owner mobile phone as it has all yes
data = data.drop(['Owned_Mobile_Phone','Applicant_ID'], axis = 1)


# In[28]:


# check correlation between variables
correlation = round(data.corr(),2)
correlation


# In[29]:


# plot correaltion

plt.figure(figsize = (20,20))
sns.heatmap(correlation, annot = True, cmap="Greens")


# In[93]:


char_col = ['Applicant_Gender','Income_Type','Education_Type','Family_Status','Housing_Type','Job_Title']
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

sns.histplot(data, x="Applicant_Gender", kde=False, color="skyblue", ax=axs[0, 0])
sns.histplot(data, x="Income_Type", kde=False, color="olive", ax=axs[0, 1])
sns.histplot(data, x="Education_Type", kde=False, color="gold", ax=axs[1, 0])
sns.histplot(data, x="Family_Status", kde=False, color="gold", ax=axs[1, 1])
sns.histplot(data, x="Housing_Type", kde=False, color="gold", ax=axs[2, 0])
sns.histplot(data, x="Job_Title", kde=False, color="gold", ax=axs[2, 1])


plt.show()


# In[30]:


# create a list of repeated values variable
col = [ 'Applicant_Gender', 'Owned_Car', 'Owned_Realty',
       'Total_Children', 'Income_Type', 'Education_Type','Family_Status', 'Housing_Type',
       'Owned_Work_Phone', 'Owned_Phone', 'Owned_Email', 'Job_Title',
       'Total_Family_Members']


# In[31]:


# Replacing categorical columns with a one-hot vector
for column in col:
    cols = pd.get_dummies(data[column], prefix= column)
    data[cols.columns] = cols
    data.drop(column, axis = 1, inplace = True)
data.shape


# In[32]:


data.head()


# In[33]:


y = data['Status']
x = data.drop(['Status'], axis = 1)


# In[34]:


print(x.shape)
print(y.shape)


# In[35]:


#import statsmodels.api as sm
# logit_model=sm.Logit(y, x)
# result=logit_model.fit()
# print(result.summary())


# In[38]:


get_ipython().system('pip install imblearn')


# In[39]:


from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)


# In[40]:


from collections import Counter
print('Resampled dataset shape %s' % Counter(y_res))


# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=0)


# In[42]:


x_train.shape, y_train.shape


# In[43]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[44]:


ypred = model.predict(x_test)
print('Test Accuracy of the model is : ', round(model.score(x_test, y_test),4)*100)
print('Train Accuracy of the model is : ', round(model.score(x_train, y_train),4)*100)


# In[45]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, ypred)
print(confusion_matrix)


# In[46]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ypred))


# In[47]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# 

# In[66]:


x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size = 1/5)


# In[67]:


pred_result = model.predict(x_test)
print(pred_result)


# In[68]:


filename = 'credit_card_approval'
pickle.dump(model, open(filename, 'wb'))


# In[63]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[59]:





# In[60]:





# In[3]:


pd.__version__


# In[69]:


np.__version__


# In[73]:


import sklearn as sklearn


# In[71]:


sns.__version__


# In[74]:


sklearn.__version__


# In[ ]:




