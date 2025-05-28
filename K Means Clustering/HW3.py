import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

class KMeansCluster:
    def __init__(self, data):
        self.data = data

    def cluster(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.data)
        return kmeans

class ClusterVisualization:
    def __init__(self, data, clustering_model):
        self.data = data
        self.model = clustering_model
        self.cluster_number = clustering_model.n_clusters
        self.clusters = []
        for i in range(self.cluster_number):
            self.clusters.append(data.loc[clustering_model.labels_ == i, :])

    def scatter_plot(self, feat1, feat2):
        colors = ['red', 'green', 'blue', 'black']
        for i in range(self.cluster_number):
            plt.scatter(self.clusters[i].loc[:, feat1], self.clusters[i].loc[:, feat2], label=f'Cluster {i+1}', color=colors[i])
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.legend()
        plt.title(f'Clusters based on {feat1} and {feat2}')
        plt.show()

file_path = 'market_ds.csv'
df = pd.read_csv(file_path)

if df.isnull().sum().any():
    print("Missing values found, filling with mean values...")
    df.fillna(df.mean(), inplace=True)

print("Dataset after pre-processing:")
print(df.head())

df_standardized = (df - df.mean()) / df.std()

kmeans_clusterer = KMeansCluster(df_standardized)
kmeans_model = kmeans_clusterer.cluster(n_clusters=3)

cluster_vis = ClusterVisualization(data=df_standardized, clustering_model=kmeans_model)
cluster_vis.scatter_plot(feat1='Income', feat2='Spending')

# Elbow method to find the optimal number of clusters
inertias = []
for i in range(1, 11):
    kmeans_model = KMeans(n_clusters=i, random_state=42).fit(df_standardized)
    inertias.append(kmeans_model.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

# Hierarchical clustering (Agglomerative Clustering) - Dendrogram
linkage_data = linkage(df_standardized, method='single', metric='euclidean')
plt.figure(figsize=(10, 6))
dendrogram(linkage_data, truncate_mode='level', p=7)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()