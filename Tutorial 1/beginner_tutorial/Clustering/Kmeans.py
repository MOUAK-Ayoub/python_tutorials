import commun
from sklearn.preprocessing import StandardScaler
import numpy as np
import KmeansRandomData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3D(X):
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    ax.set_xlabel('Education')
    ax.set_ylabel('Age')
    ax.set_zlabel('Income')
    ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))
    plt.show()

def plot2D(X):
    area = np.pi * (X[:, 1]) ** 2
    plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
    plt.xlabel('Age', fontsize=18)
    plt.ylabel('Income', fontsize=16)
    plt.show()

if  __name__ == '__main__':
    data = commun.readCsvReturnDF('Cust_Segmentation.csv')
    data = data.drop('Address', axis=1)
    X = data.values[:, 1:]
    X = np.nan_to_num(X)
    Clus_dataSet = StandardScaler().fit_transform(X)
    labels, centers = KmeansRandomData.KmeansClustering(Clus_dataSet, 3)
    data["Clus_km"] = labels
    print centers
    print data.groupby('Clus_km').mean()
    plot2D(X)

