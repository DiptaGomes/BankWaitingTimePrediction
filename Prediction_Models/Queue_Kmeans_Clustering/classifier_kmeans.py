import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date
from sklearn.cluster import KMeans 


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
kmeans1 = KMeans(n_clusters=10)  
kmeans1.fit(c,servicetime_column)
predicted_service_time = kmeans1.predict([[position, 1]])

# predict waiting time based on position and predicted service time
y = np.array(x['waiting_time'])
x = np.column_stack((x.Position, x.Service_time))
x.shape
kmeans2 = KMeans(n_clusters=10)  
kmeans2.fit(x,y)
prediction = kmeans2.predict([[position, predicted_service_time]])
print(prediction)

# seperate coulumns for visualization
x_axis_position = dataset.iloc[:,0].values
y_axis_service = dataset.iloc[:,3].values
y_axis_waiting = dataset.iloc[:,2].values

# #line graph for service time
# plt.plot(x_axis_position, y_axis_service, color='orange')
# plt.xlabel('Position')
# plt.ylabel('Service')
# plt.title('Service time in dataset with respect to position')
# plt.show()

#bar plots
plt.bar(x_axis_position, y_axis_service, color='orange')
plt.xlabel('Position')
plt.ylabel('Service Time')
plt.title('Service time in dataset with respect to position')
plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime
# from datetime import date


# x = pd.read_csv('dataset.csv')
# a=np.array(x)

# y = np.array(x['waiting_time'])

# z = np.array(x['Service_time'])


# x = np.column_stack((x.Position, x.Service_time))
# x.shape

# from sklearn.cluster import KMeans  
# kmeans = KMeans(n_clusters=3)  
# kmeans.fit(x,y)

# kmeans = KMeans(n_clusters=3)  
# kmeans.fit(y,z)

# prediction1 = kmeans.predict([[1]])
# prediction2 = kmeans.predict([[9]])
# print(prediction1)
# print(prediction2)
