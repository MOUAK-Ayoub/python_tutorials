import numpy as np
import  machine_deep_learning.neural_networks.commun as cm
import matplotlib.pyplot as plt


N = 300
K = 3
hidden_layer_size = 100
gradient_step = 1e-2
reg = 0.05
number_iteration = 1000


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

def loss_function(output_3, target,net_parameters):
    N = output_3.shape[0]

    data_unnormalized = np.exp(output_3)
    probability = data_unnormalized/ np.sum(data_unnormalized, axis = 1, keepdims= True)
    point_loss = -np.log(probability[range(N), target])
    loss = np.mean(point_loss)
    reg_loss = 0.5*reg*( np.sum(net_parameters[0]**2) + np.sum(net_parameters[1]**2))
    loss += reg_loss
    return loss, probability

def gradient_parameters(data, target, net_parameters, net_outputs, probability):
    net_parameters_updated = []

    N = data.shape[0]
    W1 = net_parameters[0]
    W2 = net_parameters[2]
    b1 = net_parameters[1]
    b2 = net_parameters[3]

    output_2 = net_outputs[1]

    dRelu = cm.relu_derivative(net_outputs[0])
    dProb = probability
    dProb[range(N), target] -= 1
    dProb /= N
    dHidden = np.dot(dProb,W2.T)

    dW1 = np.dot(data.T,dRelu*dHidden)
    db1 = np.sum(dRelu*dHidden, axis=0)
    dW2 = np.dot(output_2.T, dProb)
    db2 = np.sum(dProb, axis=0)

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
        loss, probability = loss_function(net_output[2], target, net_parameters)
        loss_array.append(loss)
        net_parameters = gradient_parameters(data, target, net_parameters, net_output, probability.copy())

    plt.plot(range(number_iteration), loss_array)
    plt.show()

    print('Min train loss {0}'.format(np.min(loss)))
    return probability, net_parameters


if __name__ == '__main__':

    data, result = cm.generate_data(N, K)
    attribut_train, attribut_test, result_train, result_test = cm.split_data(data, result, 0.2)

    print('---------------------------------Train accuracy --------------------------------------------')

    probability, net_parameters = optmize_parameters(attribut_train,result_train)
    cm.accuracy(attribut_train, result_train, probability )

    net_output = neural_net(attribut_test, net_parameters)
    loss, probability_test = loss_function(net_output[2], result_test, net_parameters)
    print('---------------------------------Test accuracy --------------------------------------------')
    print('Loss_test is {0}'.format(loss))
    cm.accuracy(attribut_test, result_test, probability_test)


