import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

class Cluster_Visualization:
    def __init__(self, data, clustering_model):
        self.__data = data
        self.__model = clustering_model
        self.__cluster_number = clustering_model.n_clusters
        self.__clusters = []
        for i in range(self.__cluster_number):
            self.__clusters.append( data.loc[model.labels_ == i,:])
    def scatter_plot(self, feat1 , feat2):
        colors= ['red','green','blue','black']
        for i in range(self.__cluster_number):
            plt.scatter(self.__clusters[i].loc[:, feat1], self.__clusters[i].loc[:, feat2], color=colors[i])
        plt.show()

#Input Dateset
train_feat = pd.read_csv("Nutritions.csv")
##Encoding Categorical Variables
train_feat= pd.get_dummies(train_feat, dtype='int')
print(train_feat)
train_feat = (train_feat - train_feat.mean()) / train_feat.std()
#kmeans
model = KMeans(n_clusters=3)
model.fit(train_feat)
clusters_vis= Cluster_Visualization(data=train_feat, clustering_model=model)
clusters_vis.scatter_plot(feat1='calories', feat2='fat')
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i).fit(train_feat)
    inertias.append(kmeans.inertia_) #inertia: Sum of squared distances of data points from their cluster centers (WCSS)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.show()
#Agnes
linkage_data = linkage(train_feat, method='single', metric='euclidean')
dendrogram(linkage_data, truncate_mode = 'level' ,p=7 )
plt.show()