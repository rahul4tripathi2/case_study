'''------------------- importing essential libraries -------------------'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

'''------------------- steps to create and run ml models -------------------'''
'''
1. Data preprocessing.
2. fit,tune & apply ml models.
3. Evaluate the model.
'''
'''------------------- starting the steps-------------------'''

# calculating the duration of code
import time
start_time = time.time()

'''=================================================================================================='''
'''-------------------------------- Step 1 - Data preprocessing -------------------------------'''
'''==================================================================================================''' 

'''------------------- loading banking dataset -------------------'''

# reading entire rows of the dataset
dummy_df = pd.read_csv("bank_data.csv", na_values =['NA'])
temp = dummy_df.columns.values[0]

columns_name = temp.split(';')
data = dummy_df.values

contacts = list()
for element in data:
    contact = element[0].split(';')
    contacts.append(contact)

contact_df = pd.DataFrame(contacts,columns = columns_name)


'''------------------- handling categorical variables -------------------'''

def preprocessor(df):
    res_df = df.copy()
    le = preprocessing.LabelEncoder() # LabelEncoder to deal with categorical variables
    
    # converting all categorical columns to numbers
    res_df['"job"'] = le.fit_transform(res_df['"job"'])
    res_df['"marital"'] = le.fit_transform(res_df['"marital"'])
    res_df['"education"'] = le.fit_transform(res_df['"education"'])
    res_df['"default"'] = le.fit_transform(res_df['"default"'])
    res_df['"housing"'] = le.fit_transform(res_df['"housing"'])
    res_df['"month"'] = le.fit_transform(res_df['"month"'])
    res_df['"loan"'] = le.fit_transform(res_df['"loan"'])
    res_df['"contact"'] = le.fit_transform(res_df['"contact"'])
    res_df['"day_of_week"'] = le.fit_transform(res_df['"day_of_week"'])
    res_df['"poutcome"'] = le.fit_transform(res_df['"poutcome"'])
    res_df['"y"'] = le.fit_transform(res_df['"y"'])
    return res_df

# passing entire dataframe to preprocessor function to get encoded dataframe
encoded_df = preprocessor(contact_df)

# dropping deposit(y) column as this is highly corelated with duration columns
x = encoded_df.drop(['"y"'],axis =1).values
y = encoded_df['"y"'].values

# distribute the dataset into train & test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.5)

'''=================================================================================================='''
'''-------------------------------- Step 2&3 - Model Fitting and Evaluation -------------------------------'''
'''=================================================================================================='''


'''------------------- model fitting -------------------'''

#decision tree
# selection information gain (~entropy) as hyperparameter to get required root node
model_dt = DecisionTreeClassifier(max_depth = 8, criterion ="entropy")
model_dt.fit(x_train, y_train)
y_pred_dt = model_dt.predict_proba(x_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

#adaboost
model_ada = AdaBoostClassifier(n_estimators = 120)
model_ada.fit(x_train, y_train)
y_pred_ada = model_ada.predict_proba(x_test)[:, 1]
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)


#SVM
# selected linear svm as a classifier
model_svc = LinearSVC(random_state=0)
model_svc.fit(x_train, y_train)
y_pred_svc = model_svc.fit(x_train, y_train).decision_function(x_test)
fpr_svc, tpr_svc, _= roc_curve(y_test, y_pred_svc)
roc_auc_svc = auc(fpr_svc, tpr_svc)


#bayes
# selected gaussian naive bayes as classifier
model_bayes = GaussianNB()
model_bayes.fit(x_train, y_train)
y_pred_bayes = model_bayes.predict(x_test)
fpr_bayes, tpr_bayes, _= roc_curve(y_test, y_pred_bayes)
roc_auc_bayes = auc(fpr_bayes, tpr_bayes)

#random forest
# selected max_depth 10 and n_estimator = 120 as hyperparameter of random forest
model_rf = RandomForestClassifier(max_depth = 8, n_estimators = 120)
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)


'''------------------- ROC CURVE -------------------'''

# Displaying ROC Curve
#ROC Curve
plt.figure(1)
lw = 2
plt.plot(fpr_bayes, tpr_bayes, color='blue',
         lw=lw, label='naive_bayes(AUC = %0.2f)' % roc_auc_bayes)
plt.plot(fpr_svc, tpr_svc, color='darkgreen',
         lw=lw, label='svm(AUC = %0.2f)' % roc_auc_svc)
plt.plot(fpr_ada, tpr_ada, color='red',
         lw=lw, label='AdaBoost(AUC = %0.2f)' % roc_auc_ada)
plt.plot(fpr_rf, tpr_rf, color='darkorange',
         lw=lw, label='Random Forest(AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_dt, tpr_dt, color='green',
         lw=lw, label='Decision Tree(AUC = %0.2f)' % roc_auc_dt)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic(ROC_curve)')
plt.legend(loc="lower right")


'''=================================================================================================='''
'''-------------------------------- Checking important features in dataset -------------------------------'''
'''=================================================================================================='''

# selecting important features in the dataset
importances = model_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(2)
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="cyan", align="center")
plt.xticks(range(x_train.shape[1]), indices)
plt.xlim([-1, x_train.shape[1]])
plt.show()

print "Important five features are : "
print ">>>>>>>>>>>>"
print columns_name[10]
print columns_name[18]
print columns_name[19]
print columns_name[12]
print columns_name[14]

print ">>>>>>>>>>>>"

print ("Total execution time is %s seconds " % (time.time() - start_time) )


