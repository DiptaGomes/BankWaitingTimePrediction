import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')
predicted = pd.read_csv('predicted_dataset.csv')

serviceTime_column = dataset.iloc[:,3].values
waitingTime_column = dataset.iloc[:,2].values
position_column = dataset.iloc[:,0].values
Predicted_position_column = predicted.iloc[:,0].values
predicted_waitingTime_KNN = predicted.iloc[:,2].values
predicted_waitingTime_KMeans = predicted.iloc[:,3].values
predicted_waitingTime_SVR = predicted.iloc[:,1].values

#Visualization
_, ax = plt.subplots()

#Barplot service time with respect to position
# ax.bar(position_column, serviceTime_column, color = '#539caf', align = 'center')
# ax.set_ylabel("Service Time")
# ax.set_xlabel("Position")
# ax.set_title("Service time in dataset with respect to Position ")

#Barplot waiting time with respect to position
# ax.bar(position_column, waitingTime_column, color = '#539caf', align = 'center')
# ax.set_ylabel("Waiting Time")
# ax.set_xlabel("Position")
# ax.set_title("Waiting time in dataset with respect to Position ")

#scatter plot service time with respect to waiting Time
# ax.scatter(serviceTime_column, waitingTime_column, s = 10, color = "red", alpha = 0.75)
# ax.set_title("Service TIme with respect to Waiting time")
# ax.set_xlabel("Service Time")
# ax.set_ylabel("Waiting Time")

#Barplot Predicted waiting timein SVR
# ax.bar(Predicted_position_column, predicted_waitingTime_SVR, color = '#539caf', align = 'center')
# ax.set_ylabel("Predicted waiting Time")
# ax.set_xlabel("Position")
# ax.set_title("Predicted Waiting time using SVR ")

#Barplot Predicted waiting timein KMeans
# ax.bar(Predicted_position_column, predicted_waitingTime_KMeans, color = '#539caf', align = 'center')
# ax.set_ylabel("Predicted waiting Time")
# ax.set_xlabel("Position")
# ax.set_title("Predicted Waiting time using KMeans ")

#Barplot Predicted waiting timein KNN
ax.bar(Predicted_position_column, predicted_waitingTime_KNN, color = '#539caf', align = 'center')
ax.set_ylabel("Predicted waiting Time")
ax.set_xlabel("Position")
ax.set_title("Predicted Waiting time using KNN ")

#Line Plots Predicted waiting timein KNN
ax.bar(Predicted_position_column, predicted_waitingTime_KNN, color = '#539caf', align = 'center')
ax.set_ylabel("Predicted waiting Time")
ax.set_xlabel("Position")
ax.set_title("Predicted Waiting time using KNN ")


#Output
plt.show()