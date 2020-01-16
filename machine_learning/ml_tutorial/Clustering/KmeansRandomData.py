import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


def generateRandomDate():
    np.random.seed(0)
    X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    #plt.show()
    return X,y

def KmeansClustering(data,nbClusters):
    k_means = KMeans(init="k-means++", n_clusters= nbClusters, n_init=12)
    k_means.fit(data)

    return k_means.labels_,k_means.cluster_centers_

def plotCluster(k_means_labels, k_means_cluster_centers):
    fig = plt.figure(figsize=(6, 4))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
    ax = fig.add_subplot(1, 1, 1)

    for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

        my_members = (k_means_labels == k)
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()

if __name__ == '__main__':
    X,y=generateRandomDate()
    k_means_labels, k_means_cluster_centers = KmeansClustering(X, 4)
    plotCluster(k_means_labels, k_means_cluster_centers)
