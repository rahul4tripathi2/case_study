# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 13:43:20 2017

@author: suryaprakash.s.singh
"""

import shelve, pandas as pd, datetime, calendar
from dateutil import parser as dp
shelf = shelve.open('HnM_Data_Shelf')
finalDf = shelf['finalDf']
newFinalDf = shelf['newFinalDf']
subDf = shelf['subDf']
dfInfo = pd.DataFrame()
dfInfo['Columns'] = list(newFinalDf.columns.values)
dfInfo['NullCount'] = list(newFinalDf.isnull().sum())
dfInfo['ColDataType'] = list(newFinalDf.dtypes)
categoryValdict = {}
for i in range(len(dfInfo)):   
  categoryValdict[dfInfo['Columns'][i]] = newFinalDf[dfInfo['Columns'][i]].value_counts().to_dict()
categoryValdict.pop('Order Creation Time', None)
categoryValdict.pop('Return Creatio Time', None)
shelf.close()

finalDf.to_csv('HnM_FinalDF.csv', encoding='utf-8', na_rep='')


usableRecords = shelf['usableRecords']
#subDf = usableRecords.iloc[:, [0,1,3,5,6,10,13,14,15,16,18,19,20,22,24]]
#newFinalDf = subDf.iloc[:, [0,1,2,3,4,5,7,8,10,11]]

#newFinalDf.to_csv('HnM_FinalDF.csv', encoding='utf-8', na_rep='')
OrderMonth = []
for i in range(len(newFinalDf)):
  OrderMonth.append(calendar.month_name[datetime.datetime.strptime(newFinalDf['Order Creation Time'][i].split('-')[1], '%b').month])

ReturnMonth = []
for i in range(len(newFinalDf)):
  if newFinalDf['Returned'][i] == 1:
    OrderMonth.append(calendar.month_name[datetime.datetime.strptime(newFinalDf['Return Creatio Time'][5].split('-')[1], '%b').month])



'''=================================================================================================='''
'''-------------------------------- Preprocessing Categorical Data -------------------------------'''
'''=================================================================================================='''
#import UtilityFunctions as uf
#shelf = shelve.open('HnM_Data_Shelf')
#subDf = shelf['subDf']
#shelf.close()
#finalDf = uf.multiCategory2LabelEncoder(subDf)
X = finalDf.iloc[:, [0,1,2,3,4,5,7]].values
y = finalDf.iloc[:, [6]].values


'''=================================================================================================='''
'''-------------------------------- Model Fitting -------------------------------'''
'''=================================================================================================='''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


'''------------------- Random Forest Model Fitting -------------------'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = rf_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
rf = confusion_matrix(y_test, y_pred)
print rf

rf_classifier.feature_importances_

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
# Visualising the Test set results


'''------------------- Decision Tree Model Fitting -------------------'''
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dt_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
dt = confusion_matrix(y_test, y_pred)
print dt

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

dt_classifier.feature_importances_


'''------------------- Naive Bayes Model Fitting -------------------'''
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nb_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
dt = confusion_matrix(y_test, y_pred)
print dt

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


'''------------------- Support Vector Machine (SVM) Fitting -------------------'''
# Fitting SVM to the Training set
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'linear', random_state = 0)
svc_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = svc_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)