import numpy as np
from commun import generate_data

K= 3 # number of clusters
N = 100 # number of points per class
reg = 0.5


def loss_array(data):
    W = 0.1 * np.random.randn( 2, K)
    b = np.zeros((1,K))
    data_weighted = np.dot(data, W)+ b
    data_unnormalized = np.exp(data_weighted)
    probability = data_unnormalized/ np.sum(data_unnormalized, axis = 1, keepdims= True)
    point_loss = -np.log(probability)
    data_loss = np.mean(point_loss)
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss


if __name__ == '__main__':

    data = generate_data(N, K)
    loss_array(data)
