import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

# Read data from file
customer_data = pd.read_csv('shopping_data.csv')
print(customer_data.shape)
print(customer_data.head())
data = customer_data.iloc[:, 3:5].values

from sklearn.cluster import AgglomerativeClustering

#cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
#cluster.fit_predict(data)


linked = linkage(data, 'ward')
#set labellist size according to linked size, data is 200,5
labelList = range(1, len(data)+1)
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()