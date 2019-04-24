import numpy as np
import pandas as pd
from matplotlib import cm

def success_ratio(cm):
    total = cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]
    return 100*(cm[0][0] + cm[1][1]) / total


dataset = pd.read_csv('dataset.csv')

X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,len(dataset.iloc[0])-1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#fitting the classifier to the training set

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7)
classifier.fit(X_train, y_train)

#predicting the results on the training set
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("Training set confusion matrix : \n"+str(cm_train))
print("Success ratio on training set : "+str(success_ratio(cm=cm_train))+"%")
print("Test set confusion matrix : \n"+str(cm_test))
print("Success ratio on test set : "+str(success_ratio(cm=cm_test))+"%")