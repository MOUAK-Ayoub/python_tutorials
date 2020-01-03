import numpy as np
from commun import generate_data, split_data
import commun
import matplotlib.pyplot as plt

K= 3 # number of clusters
reg = 0
gradient_step = 1e-2
number_iteration = 10000



def loss_function(W, b, data, target):
    N = data.shape[0]
    data_weighted = np.dot(data, W)+ b
    data_unnormalized = np.exp(data_weighted)
    probability = data_unnormalized/ np.sum(data_unnormalized, axis = 1, keepdims= True)
    point_loss = -np.log(probability[range(N),target])
    data_loss = np.mean(point_loss)
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss

    return loss, probability

def gradient_parameters(W, b,data, target, probability):
    N = data.shape[0]
    dProb = probability
    dProb[range(N), target] -= 1
    dProb /= data.shape[0]
    dW = np.dot(data.T, dProb) + reg * W
    db = np.sum(dProb, axis=0)
    W -= gradient_step * dW
    b -= gradient_step * db

    return W, b

def optmize_parameters(W, b, data, target):
    loss_array=[]

    for i in range(number_iteration):
        loss, probability = loss_function(W, b, data, target)
        loss_array.append(loss)
        W, b = gradient_parameters(W,b, data, target, np.copy(probability))

    plt.plot(range(number_iteration), loss_array)
    plt.show()
    predicted = np.argmax(probability, axis=1)


    return predicted, W, b





if __name__ == '__main__':
    N = 3000  # number of points per class

    W =   0.01 *  np.random.randn(2, K)
    b =  0.01 * np.random.randn(1, K)
    data, target = generate_data(N, K)

    attribut_train,attribut_test,result_train,result_test = split_data(data,target, testSize=0.3)
    predicted_train, W, b = optmize_parameters(W, b, attribut_train, result_train)
    commun.accuracy(attribut_train,result_train,predicted_train )

    loss_test, probability_test = loss_function(W, b, attribut_test, result_test)
    print(loss_test)
    predicted_test = np.argmax(probability_test, axis=1)
    commun.accuracy(attribut_test, result_test, predicted_test)
