import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import preprocessing


# Importing dataset and examining it
dataset = pd.read_csv("E-Shop.csv")
pd.set_option('display.max_columns', None) # Will ensure that all columns are displayed
print(dataset.head())
print(dataset.shape)
# print(dataset.info())
# print(dataset.describe())

 #Converting Categorical features into Numerical features
def converter(column):
    if column =='TRUE':
        return 1
    else:
        return 0
le = preprocessing.LabelEncoder()
dataset['Weekend'] = le.fit_transform(dataset['Weekend'])
le = preprocessing.LabelEncoder()
dataset['Transaction'] = le.fit_transform(dataset['Transaction'])
# We can also write - dataset['OverTime'] = dataset['OverTime'].map({'Yes':1, 'No':0})
print(dataset.info())


categorical_features = ['Month', 'VisitorType']
final_data = pd.get_dummies(dataset, columns = categorical_features)
# print(final_data.info())
# print(final_data.head(2))

# Dividing dataset into label and feature sets
X = final_data.drop('Transaction', axis = 1) # Features
Y = final_data['Transaction'] # Labels
print(np.unique(Y))

print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

## Normalizing numerical features so that each feature has mean 0 and variance 1
#feature_scaler = StandardScaler()
#X_scaled = feature_scaler.fit_transform(X)

## Dividing dataset into training and test sets
#X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

#print(X_train.shape)
#print(X_test.shape)
###########################################################################################################################
## Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
#print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

#smote = SMOTE(random_state = 101)
#print(np.unique(Y))
#X_train,Y_train = smote.fit_sample(X,Y)

#print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())

##unique, counts = np.unique(Y, return_counts=True)
##plt.figure(figsize=(8,6))
##sns.barplot(x=unique, y=counts)
##plt.title('Number of transaction for each class before applying SMOTE')
##plt.ylabel('Count')
##plt.xlabel('Class (0:, 1:)')

##unique, counts = np.unique(Y_train, return_counts=True)
##plt.figure(figsize=(8,6))
##sns.barplot(x=unique, y=counts)
##plt.title('Number of transaction for each class AFTER applying SMOTE')
##plt.ylabel('Count')
##plt.xlabel('Class (0:, 1:)')


## Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
#rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
#grid_param = {'n_estimators': [200, 250, 300, 350, 400, 450]}

#gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='recall', cv=5)

#"""
#In the above GridSearchCV(), scoring parameter should be set as follows:
#scoring = 'accuracy' when you want to maximize prediction accuracy
#scoring = 'recall' when you want to minimize false negatives
#scoring = 'precision' when you want to minimize false positives
#scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
#"""
#score_array=[]
#gd_sr.fit(X_train, Y_train)

#best_parameters = gd_sr.best_params_
#print(best_parameters)

#best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
#print(best_result)
#score_array.append(best_result)

## Building random forest using the tuned parameter
#rfc = RandomForestClassifier(n_estimators=400, criterion='entropy', max_features='auto', random_state=1)
#rfc.fit(X_train,Y_train)
#featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
#print(featimp)

#Y_pred = rfc.predict(X_test)
#print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

#confusion_mat = metrics.confusion_matrix(Y_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(confusion_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', confusion_mat)
#print('TP: ', confusion_mat[1,1])
#print('TN: ', confusion_mat[0,0])
#print('FP: ', confusion_mat[0,1])
#print('FN: ', confusion_mat[1,0])   


### Selecting features with higher sifnificance and building random forest
#X1 = final_data[['PageValue', 'ExitRate', 'Administrative','ProductRelated_Duration','ProductRelated']]

###feature_scaler = StandardScaler()
#X1_scaled = feature_scaler.fit_transform(X1)
#X1_train, X1_test, Y1_train, Y1_test = train_test_split( X1_scaled, Y, test_size = 0.3, random_state = 100)

#smote = SMOTE(random_state = 101)
#X1_train,Y1_train = smote.fit_sample(X1_train,Y1_train)


###rfc = RandomForestClassifier(n_estimators=400, criterion='entropy', max_features='auto', random_state=1)
#rfc.fit(X1_train,Y1_train)

#Y_pred = rfc.predict(X1_test)
#print('Classification report: \n', metrics.classification_report(Y1_test, Y_pred))

#confusion_mat = metrics.confusion_matrix(Y1_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(confusion_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', confusion_mat)
#print('TP: ', confusion_mat[1,1])
#print('TN: ', confusion_mat[0,0])
#print('FP: ', confusion_mat[0,1])
#print('FN: ', confusion_mat[1,0]) 

###########################################################################################################################

### Tuning the AdaBoost parameter 'n_estimators' and implementing cross-validation using Grid Search
#abc = AdaBoostClassifier(random_state=1)
#grid_param = {'n_estimators': [5,10,20,30,40,50]}

#gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='recall', cv=5)

###"""
###In the above GridSearchCV(), scoring parameter should be set as follows:
###scoring = 'accuracy' when you want to maximize prediction accuracy
###scoring = 'recall' when you want to minimize false negatives
###scoring = 'precision' when you want to minimize false positives
###scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
###"""

#gd_sr.fit(X_train, Y_train)

#best_parameters = gd_sr.best_params_
#print(best_parameters)

#best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
#print(best_result)
#score_array.append(best_result)

### Building AdaBoost using the tuned parameter
#abc = AdaBoostClassifier(n_estimators=20, random_state=1)
#abc.fit(X_train,Y_train)
#featimp = pd.Series(abc.feature_importances_, index=list(X)).sort_values(ascending=False)
#print(featimp)

#Y_pred = abc.predict(X_test)
#print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

#confusion_mat = metrics.confusion_matrix(Y_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(confusion_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', confusion_mat)
#print('TP: ', confusion_mat[1,1])
#print('TN: ', confusion_mat[0,0])
#print('FP: ', confusion_mat[0,1])
#print('FN: ', confusion_mat[1,0])


###########################################################################################################################

### Tuning the Gradent Boost parameter 'n_estimators' and implementing cross-validation using Grid Search
#gbc = GradientBoostingClassifier(random_state=1)
#grid_param = {'n_estimators': [10,20,30,40,50], 'max_depth' : [5,6,7,8,9,10,11,12], 'max_leaf_nodes': [8,12,16,20,24,28,32]}

#gd_sr = GridSearchCV(estimator=gbc, param_grid=grid_param, scoring='recall', cv=5)

###"""
###In the above GridSearchCV(), scoring parameter should be set as follows:
###scoring = 'accuracy' when you want to maximize prediction accuracy
###scoring = 'recall' when you want to minimize false negatives
###scoring = 'precision' when you want to minimize false positives
###scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
###"""

#gd_sr.fit(X_train, Y_train)

#best_parameters = gd_sr.best_params_
#print(best_parameters)

#best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
#print(best_result)
#score_array.append(best_result)

### Building Gradient Boost using the tuned parameter
#gbc = GradientBoostingClassifier(n_estimators=30, max_depth=10, max_leaf_nodes=32, random_state=1)
#gbc.fit(X_train,Y_train)
#featimp = pd.Series(gbc.feature_importances_, index=list(X)).sort_values(ascending=False)
#print(featimp)

#Y_pred = gbc.predict(X_test)
#print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

#confusion_mat = metrics.confusion_matrix(Y_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(confusion_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', confusion_mat)
#print('TP: ', confusion_mat[1,1])
#print('TN: ', confusion_mat[0,0])
#print('FP: ', confusion_mat[0,1])
#print('FN: ', confusion_mat[1,0])






