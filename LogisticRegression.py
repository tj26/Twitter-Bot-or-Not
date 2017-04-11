#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 00:13:39 2017

@author: tanyajha
"""

import pandas as pd
import numpy as np


names = ['id','id_str','screen_name','location','url','followers_count','friends_count','listedcount','created_at','favourites_count','verified','statuses_count','language','status','default_profile','default_profile_image','has_extended_profile','name','bot']

data1 = pd.read_csv('bots_data.csv',header = 0, delim_whitespace = False,encoding='ISO-8859-1', names=names,na_values = 'NaN')

data2 = pd.read_csv('nonbots_data.csv',header = 0, delim_whitespace = False,encoding='ISO-8859-1', names=names,na_values ='NaN')

frames = [data1,data2]

#Contains the merged data of bots and not bots 
data = pd.concat(frames)

         
X = data.loc[:,['followers_count','friends_count','listedcount','favourites_count','statuses_count','verified','default_profile']]
#,'Verified','Default_profile','Verified','Default_profile_image','Has_extended_profile']]

#One hot encode categorical features verified and default_profile
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder(categorical_features=[5,6])
X = enc.fit_transform(X).toarray()

y = data.loc[:,['bot']]

#Split dataset into testing and training
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, class_weight = 'balanced')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)


from sklearn import metrics
print("Logistic Regression:",metrics.accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Cross Validation",accuracies.mean())


print("AUC",metrics.roc_auc_score(y_test, y_pred))
print("Precision: ",metrics.precision_score(y_test,y_pred))
print("Recall: ",metrics.recall_score(y_test,y_pred))
print("F1: ",metrics.f1_score(y_test,y_pred))

import matplotlib.pyplot as plt
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
