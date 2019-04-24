import numpy as np
import pandas as pd
from matplotlib import cm

# def success_ratio(cm):
#     total = cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]
#     return 100*(cm[0][0] + cm[1][1]) / total




# Cross Validation Classification Accuracy
from sklearn import model_selection
from sklearn.svm import SVR

dataframe = pd.read_csv('dataset.csv')

array = dataframe.values
X = array[:,[0,4]]
Y = array[:,6]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = svr_lin = SVR(kernel='linear')

scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
# X = dataset.iloc[:,0:3].values
# y = dataset.iloc[:,len(dataset.iloc[0])-1].values

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# #fitting the classifier to the training set
# from sklearn.svm import SVR
# classifier = svr_lin = SVR(kernel='linear')
# classifier.fit(X_train, y_train)

# #predicting the results on the training set
# y_train_pred = classifier.predict(X_train)
# y_test_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm_train = confusion_matrix(y_train, y_train_pred)
# cm_test = confusion_matrix(y_test_pred, y_test_pred)

# print("Training set confusion matrix : \n"+str(cm_train))
# print("Success ratio on training set : "+str(success_ratio(cm=cm_train))+"%")
# print("Test set confusion matrix : \n"+str(cm_test))
# print("Success ratio on test set : "+str(success_ratio(cm=cm_test))+"%")
