import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('dataset.csv')
df.shape

position = input("Where is your position? ")

x=df.iloc[:,[0, 4]].values
y=df['Service_time'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier



knn1 = KNeighborsClassifier(n_neighbors=7)
knn1.fit(X_train, y_train)
predicted_service_time = int(knn1.predict([[position, 1]]))

# print("Predicted Service Time:  ")
# print(predicted_service_time)

c=df.iloc[:,[0, 3]].values
z=df['waiting_time'].values

C_train,C_test,Z_train,Z_test = train_test_split(c,z,test_size=0.4,random_state=42, stratify=y)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(Z_train)

knn2 = KNeighborsClassifier(n_neighbors=7)
knn2.fit(C_train, encoded)
predicted_waiting_time = knn2.predict([[position, predicted_service_time]])

print("Predicted Waiting Time: ")
print(predicted_waiting_time)
