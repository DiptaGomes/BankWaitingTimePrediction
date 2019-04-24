import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVR
from sklearn.svm import SVC
from datetime import date

# read dataset file
x = pd.read_csv('dataset.csv')
dataset = pd.read_csv('dataset.csv')
a=np.array(x)

# input position 
position = input("What's your position in the queue? ")

# predict service time from dataset based on position and branch
c = np.column_stack((x.Position, x.Branches))
c.shape
servicetime_column= dataset.iloc[:,3].values
svr_lin1 = SVR(kernel='linear')
svr_lin1.fit(c, servicetime_column)
predicted_service_time = svr_lin1.predict([[position, 1]])

# predict waiting time based on position and predicted service time
y = np.array(x['waiting_time'])
x = np.column_stack((x.Position, x.Service_time))
x.shape
svr_lin = SVR(kernel='linear')
svr_lin.fit(x, y)
prediction = svr_lin.predict([[position, predicted_service_time]])
print(prediction)