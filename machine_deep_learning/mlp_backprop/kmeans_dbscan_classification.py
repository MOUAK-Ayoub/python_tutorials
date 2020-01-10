import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans
from sklearn.cluster import  DBSCAN
from commun import generate_data

def kmeans_clustering(data, nbClusters):
    k_means = KMeans(init="k-means++", n_clusters= nbClusters, n_init=12)
    k_means.fit(data)

    return k_means.labels_,k_means.cluster_centers_


def plot_clusters(data, k_labels, k_centers):
    K = len(set(k_labels))
    colors = plt.cm.Spectral(np.linspace(0, 1, K))

    for i, color in zip( range(K) , colors):

        member = data[k_labels == i]
        plt.scatter(member[:, 0], member[:, 1], s=40, cmap=color, marker='o')
        plt.scatter(k_centers[i, 0], k_centers[i, 1], s=40, cmap=color, marker='v')

    plt.show()


def dbscan_clustering(data):
    db = DBSCAN(eps=0.001, min_samples=2).fit(data)

    K = len(set(db.labels_))
    colors = plt.cm.Spectral(np.linspace(0, 1, K))

    for i, color in zip( set(db.labels_) , colors):
        color = ([0.4, 0.4, 0.4] if i == -1 else color )
        member = data[db.labels_ == i]
        plt.scatter(member[:, 0], member[:, 1], s=40, cmap=color, marker='o')

    plt.show()

if __name__ == '__main__':
    K  = 3
    data, target  = generate_data(100, 3)
    k_labels, k_centers = kmeans_clustering(data, K)
    plot_clusters(data, k_labels, k_centers)
    dbscan_clustering(data)
    correct_prediction = np.size(target[(k_labels == target)])
    print(correct_prediction)
    print(correct_prediction / float(np.size(target)) * 100)
