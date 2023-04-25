import tslearn
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("./data/dataset/A/A-1/rh_A-1.pickle",'rb') as f:
    A1 = pickle.load(f)
with open("./data/dataset/A/A-2/rh_A-2.pickle",'rb') as f:
    A2 = pickle.load(f)
with open("./data/dataset/A/A-3/rh_A-3.pickle",'rb') as f:
    A3 = pickle.load(f)

with open("./data/dataset/S/S-1/rh_S-1.pickle",'rb') as f:
    S1 = pickle.load(f)
with open("./data/dataset/S/S-2/rh_S-2.pickle",'rb') as f:
    S2 = pickle.load(f)
with open("./data/dataset/S/S-3/rh_S-3.pickle",'rb') as f:
    S3 = pickle.load(f)




#Train
datasetA1 = to_time_series(A1)
datasetA2 = to_time_series(A2)
datasetS3 = to_time_series(S3)
datasetS2 = to_time_series(S2)



#Test
datasetA3 = to_time_series(A3)
datasetS1 = to_time_series(S1)

formatted_data = to_time_series_dataset([datasetA1,datasetA2,datasetS3,datasetS2])

#print(dataset1.shape)
#print(formatted_data.shape)

x = formatted_data
y = datasetA3
n_clusters = 2
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

