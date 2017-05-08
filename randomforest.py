#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 00:03:24 2017

@author: tanyajha
"""

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

names = ['id','id_str','screen_name','location','description','url','followers_count','friends_count','listedcount','created_at','favourites_count','verified','statuses_count','language','status','default_profile','default_profile_image','has_extended_profile','name','bot']
data1 = pd.read_csv('training_data_2_csv_UTF.csv',encoding='ISO-8859-1')
data2 = pd.read_csv('nonbots_data.csv',encoding='ISO-8859-1')
frames =[data1,data2]
#Contains the merged data of bots and not bots 
data = pd.concat(frames)


data['has_extended_profile'] = data['has_extended_profile'].fillna(False)
data['verified']= data['verified'].astype(int)
data['default_profile'] = data['default_profile'].astype(int)
data['default_profile_image'] = data['default_profile_image'].astype(int)
data['has_extended_profile'] = data['has_extended_profile'].astype(int)
data['name_bot']=0
data['screen_name_bot']=0
data['description_bot']=0
data['status_bot']=0
data['location'] = pd.isnull(data.location).astype(int)
data['description'] = data['description'].fillna("")
data['status'] = data['status'].fillna("")

for i in range(0, len(data)):
    if data['screen_name'].iloc[i].lower().find('bot')== -1:
       continue
    else:
       data.iloc[i, data.columns.get_loc('screen_name_bot')] = 1

for i in range(0, len(data)):
    if data['description'].iloc[i].lower().find('bot')== -1:
       continue
    else:
       data.iloc[i, data.columns.get_loc('description_bot')] = 1

for i in range(0, len(data)):
    if data['name'].iloc[i].lower().find('bot')== -1:
       continue
    else:
       data.iloc[i, data.columns.get_loc('name_bot')] = 1

for i in range(0, len(data)):
    if data['status'].iloc[i].lower().find('bot')== -1:
       continue
    else:
       data.iloc[i, data.columns.get_loc('status_bot')] = 1                 
                   
#caculate age
dataframe = data.copy()
scratch = dataframe.copy()
scratch['age'] = 0;
a = scratch['created_at']
pd.isnull(a).values.any()

scratch['created_at'] = scratch['created_at'].map(lambda x: x.lstrip('"').rstrip('"'))
scratch['created_at'] = pd.to_datetime(scratch['created_at'], errors = 'coerce')

createddate = scratch['created_at']
nulls=(createddate.notnull()==False)
nulls.value_counts()

x = scratch[['age']].copy()
now = datetime.now()
now = now.date()
x['age'] = now - createddate
x['age'] = x['age'].astype('timedelta64[D]')

X = x['age'].copy()
X['neg'] = np.sign(X)
X['neg'].value_counts()

indx = X['neg'] > 0
indx.value_counts()
z = indx[indx == False].index.tolist()
scratch['created_at'].iloc[z]
 
x.iloc[z]
scratch['age'] = x['age']
verify = x[['age']].copy()
verify.iloc[z,0] = np.nan
verify['age'] = verify['age'].fillna(verify['age'].median())
verify['age'].iloc[z]

f, ax = plt.subplots(figsize=(10, 8))
corr = scratch.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

X = scratch.loc[:,['followers_count','friends_count','favourites_count','statuses_count','listedcount','verified','age','location','has_extended_profile','name_bot','screen_name_bot','description_bot','status_bot']]
y = scratch.loc[:,['bot']]


#Split into test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 200, criterion = 'gini')
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn import metrics
print("Random Forest:",metrics.accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Cross Validation",accuracies.mean())
print("AUC",metrics.roc_auc_score(y_test, y_pred))
print("Precision: ",metrics.precision_score(y_test,y_pred))
print("Recall: ",metrics.recall_score(y_test,y_pred))
print("F1: ",metrics.f1_score(y_test,y_pred))


from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class
 
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cf = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cf)

######################################################################################
#Test on Kaggle dataset
######################################################################################

import datetime
testbot = testbot = pd.read_csv('test_data_4_students.csv',header = 0 ,delim_whitespace = False,encoding='ISO-8859-1', names=names)
testbot['created_at'] = testbot['created_at'].fillna(datetime.datetime.now())
from datetime import datetime


dataframe = testbot.copy()
scratch = dataframe.copy()
scratch['age'] = 0;
a = scratch['created_at']
pd.isnull(a).values.any()


scratch['created_at'] = pd.to_datetime(scratch['created_at'], errors = 'coerce')

createddate = scratch['created_at']
nulls=(createddate.notnull()==False)
nulls.value_counts()

x = scratch[['age']].copy()
now = datetime.now()
now = now.date()
x['age'] = now - createddate
x['age'] = x['age'].astype('timedelta64[D]')
X = x['age'].copy()
X['neg'] = np.sign(X)
X['neg'].value_counts()

indx = X['neg'] > 0
indx.value_counts()
z = indx[indx == False].index.tolist()
scratch['created_at'].iloc[z]
 
x.iloc[z]
scratch['age'] = x['age']
scratch['age'] = x['age']
verify = x[['age']].copy()
verify.iloc[z,0] = np.nan
verify['age'] = verify['age'].fillna(verify['age'].median())
verify['age'].iloc[z]
scratch['age'] = verify['age']


scratch['name_bot']=0
scratch['screen_name_bot']=0
scratch['description_bot']=0
scratch['status_bot']=0

scratch['screen_name'].replace('None',"",inplace = True)
scratch['description'].replace('None',"",inplace = True)
scratch['name'].replace('None',"",inplace = True)
scratch['status'] = scratch['status'].fillna("")  
scratch['screen_name'] = scratch['screen_name'].fillna("")
scratch['description'] = scratch['description'].fillna("")
scratch['name'] = scratch['name'].fillna("")
scratch.fillna(0)
  
for i in range(0, len(scratch)):
     if scratch['screen_name'].iloc[i].lower().find('bot')== -1:
         continue
     else:
         scratch.iloc[i, scratch.columns.get_loc('screen_name_bot')] = 1
  
for i in range(0, len(scratch)):
     if scratch['description'].iloc[i].lower().find('bot')== -1:
         continue
     else:
         scratch.iloc[i, scratch.columns.get_loc('description_bot')] = 1
  
for i in range(0, len(scratch)):
     if scratch['name'].iloc[i].lower().find('bot')== -1:
         continue
     else:
         scratch.iloc[i, scratch.columns.get_loc('name_bot')] = 1                 

for i in range(0, len(scratch)):
     if scratch['status'].iloc[i].lower().find('bot')== -1:
         continue
     else:
         scratch.iloc[i, scratch.columns.get_loc('status_bot')] = 1                 
  
scratch['location'] = pd.isnull(scratch.location).astype(int)
         
scratch['verified'].replace('None',False,inplace = True)
scratch['verified'].replace('FALSE',False,inplace = True)
scratch['verified'].replace('TRUE',True,inplace = True)
scratch['verified']= scratch['verified'].fillna(False)
  
scratch['has_extended_profile'].replace('None',False,inplace = True)
scratch['has_extended_profile'].replace('FALSE',False,inplace = True)
scratch['has_extended_profile'].replace('TRUE',True,inplace = True)
scratch['has_extended_profile'] = scratch['has_extended_profile'].fillna(False)
  
scratch['default_profile'].replace('None',False,inplace = True)
scratch['default_profile'].replace('FALSE',False,inplace = True)
scratch['default_profile'].replace('TRUE',True,inplace = True)
scratch['default_profile'] = scratch['default_profile'].fillna(False)
  
scratch['default_profile_image'].replace('None',False,inplace = True)
scratch['default_profile_image'].replace('FALSE',False,inplace = True)
scratch['default_profile_image'].replace('TRUE',True,inplace = True)
scratch['default_profile_image'] = scratch['default_profile_image'].fillna(False)
  
scratch['default_profile_image'] = scratch['default_profile_image'].astype(int)
scratch['has_extended_profile'] = scratch['has_extended_profile'].astype(int)
scratch['verified']= scratch['verified'].astype(int)
scratch['default_profile'] = scratch['default_profile'].astype(int)
  
scratch.replace('None',0,inplace = True)
scratch.fillna(value = 0)
  
res = testbot.ix[0:574,:]
id = res.id
idx = id.astype(np.int64)
  
z= scratch.loc[:,['followers_count','friends_count','favourites_count','statuses_count','listedcount','verified','age','location','has_extended_profile','name_bot','screen_name_bot','description_bot','status_bot']]

z['followers_count'] = z['followers_count'].fillna(0).astype(int)
z['friends_count'] = z['friends_count'].fillna(0).astype(int)
z['listedcount'] = z['listedcount'].fillna(0).astype(int)
z['favourites_count'] = z['favourites_count'].fillna(0).astype(int)
z['statuses_count'] = z['statuses_count'].fillna(0).astype(int)
   
z= z.ix[0:574,:]
predicted = classifier.predict(z)
  
output = pd.DataFrame(data={"Id":idx, "bot":predicted,})
print(output)
  
output.to_csv("/Users/tanyajha/Desktop/BotOrNot/result.csv", index=False)

