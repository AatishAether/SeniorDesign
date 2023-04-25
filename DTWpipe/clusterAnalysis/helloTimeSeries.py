import tslearn
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

import pickle
import numpy as np
import matplotlib.pyplot as plt


import os
#with open("features1.pickle",'rb') as f:
#    ts1 = pickle.load(f)
#
#
#with open("features2.pickle",'rb') as f:
#    ts2 = pickle.load(f)
#
#
#with open("features3.pickle",'rb') as f:
#    ts3 = pickle.load(f)
#
#
#with open("features4.pickle",'rb') as f:
#    ts4 = pickle.load(f)
#
#with open("features5.pickle",'rb') as f:
#    ts5 = pickle.load(f)
#
#with open("features6.pickle",'rb') as f:
#    ts6 = pickle.load(f)

##Iterate through clusterAnalysis folder and load all pickle files into array
cluster = []
for filename in os.listdir("./cluster1"):
    if(filename.endswith(".pickle")):
        with open("./cluster1/"+filename,'rb') as f:
            data = pickle.load(f)
            data = to_time_series(data)
            cluster.append(data)

clusterDataset = to_time_series_dataset(cluster)

#print(clusterDataset.shape)

#print(dataset1.shape)
#print(formatted_data.shape)

x = clusterDataset
y = cluster[-1]
n_clusters = 3

print(f'x : {x}')
#km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
#labels = km.fit_predict(x)
km_bis = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw")
labels = km_bis.fit_predict(x)

fig, ax = plt.subplots()
for i in range(n_clusters):
    ax.scatter(x[labels ==i, :, 0].flatten(), x[labels ==i, :, 1].flatten(),
            label=f'Cluster {i}',alpha=0.5)
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
#print(labels)
#print(label_bis)

