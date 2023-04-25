import tslearn
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

import pickle
import numpy as np
import matplotlib.pyplot as plt


import os


cluster = []
##dir walk through data/dataset folder to load all pickle files into array
for root, dirs, files in os.walk("./data/dataset"):
    for name in files:
        if(name.endswith(".pickle")):
            if(name.startswith("pose")):
                continue
            with open(os.path.join(root,name),'rb') as f:
                data = pickle.load(f)
                if(not np.any(data)):
                    continue
                data = to_time_series(data)
                cluster.append(data)

with open("openHand.pickle",'rb') as f:
    open_hand = pickle.load(f)
    comparison = to_time_series(open_hand)
#print(clusterDataset.shape)
#for root,dirs,files in os.walk("./data/dataset/B"):
#    for name in files:
#        if(name.endswith(".pickle")):
#            if(name.startswith("pose")):
#                print(f'Ignoring {name}')
#                continue
#            with open(os.path.join(root,name),'rb') as f:
#                data = pickle.load(f)
#                if(not np.any(data)):
#                    continue
#                data = to_time_series(data)
#                comparison.append(data)
#print(dataset1.shape)
#print(formatted_data.shape)
clusterDataset = to_time_series_dataset(cluster)
comparisonDataset = to_time_series_dataset(comparison)
x = clusterDataset
y = comparisonDataset
n_clusters = 26

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

