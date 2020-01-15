import numpy as np
import machine_learning.mlp_backprop.commun as commun
import matplotlib.pyplot as plt

N = 300
K = 3
hidden_layer_size = 100
gradient_step = 1e-2
number_iteration = 10000


def init_parameters():

     net_parameters=[]
     W1 = 0.01 * np.random.randn(2,hidden_layer_size)
     b1 = 0.01 * np.random.randn(1,hidden_layer_size)
     W2 = 0.01 * np.random.randn(hidden_layer_size,K)
     b2 = 0.01 * np.random.randn(1, K)
     net_parameters.extend([W1, b1, W2, b2])

     return net_parameters

def neural_net(data,net_parameters):

    net_outputs = []
    output_1 = np.dot(data,net_parameters[0]) + net_parameters[1]
    output_2 = np.maximum(0, output_1)
    output_3 = np.dot(output_2, net_parameters[2]) + net_parameters[3]
    net_outputs.extend([output_1,output_2, output_3])

    return net_outputs

def loss_function(output_3, target):
    N = output_3.shape[0]

    data_unnormalized = np.exp(output_3)
    probability = data_unnormalized/ np.sum(data_unnormalized, axis = 1, keepdims= True)
    point_loss = -np.log(probability[range(N), target])
    loss = np.mean(point_loss)

    return loss, probability

def gradient_parameters(data, target,net_parameters, net_outputs, probability):
    net_parameters_updated = []

    N = data.shape[0]
    W1 = net_parameters[0]
    W2 = net_parameters[2]
    b1 = net_parameters[1]
    b2 = net_parameters[3]

    output_2 = net_outputs[1]

    dRelu = commun.relu_derivative(net_outputs[0])
    weight2_sum = np.sum(W2, axis=1).T
    weight2_correct = W2[range(N), target] - 1
    probability_correct = probability[range(N), target] - 1
    dProb = probability
    dProb[range(N), target] -= 1

    dW1 = np.dot(data.T*probability_correct, dRelu) * weight2_sum - np.dot(data.T , dRelu) * weight2_correct.T
    dW1 /= N

    db1 = np.sum(dRelu*probability_correct, axis=0) * weight2_sum - np.sum(dRelu, axis=0) * weight2_correct.T
    db1 /= N

    dW2 = np.dot(output_2.T, dProb)
    dW2 /= N

    db2 = np.sum(dProb, axis=0)
    db2 /= N

    W1 -= gradient_step * dW1
    b1 -= gradient_step * db1
    W2 -= gradient_step * dW2
    b2 -= gradient_step * db2

    net_parameters_updated.extend([W1, b1, W2, b2])
    return net_parameters_updated

def optmize_parameters(data, target):
    loss_array=[]
    net_parameters = init_parameters()

    for i in range(number_iteration):
        net_output = neural_net(data, net_parameters)
        loss, probability = loss_function(net_output[2], target)
        loss_array.append(loss)
        net_parameters = gradient_parameters(data, target, net_parameters, net_output, probability)

    plt.plot(range(number_iteration), loss_array)
    plt.show()
    predicted = np.argmax(probability, axis=1)

    return predicted, net_parameters

if __name__ == '__main__':

    data, result = commun.generate_data(N, K)

    attribut_train, attribut_test, result_train, result_test = commun.split_data(data, result, 0.2)

    predicted, net_parameters = optmize_parameters(attribut_train,result_train)
    commun.accuracy(attribut_train, result_train, predicted)

    net_output = neural_net(attribut_test, net_parameters)
    loss, probability_test = loss_function(net_output[2], result_test)
    print(loss)
    predicted_test = np.argmax(probability_test, axis=1)
    commun.accuracy(attribut_test, result_test, predicted_test)
