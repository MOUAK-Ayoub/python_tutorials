import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_data(N, K):
    data = np.zeros((N * K, 2))
    target = np.zeros(N*K, dtype='uint8')
    radius = np.linspace(0.0, 1.0, N)
    colors = plt.cm.Spectral(np.linspace(0, 1, K))
    fig = plt.figure()

    for i, color in zip(range(K), colors):
        theta = np.linspace(120.0 *(i-1), 230.0, N) * np.pi / 180.0 + np.random.rand(N)*0.1
        data[i * N: (i + 1) * N, 0] = radius * np.cos(theta)
        data[i * N: (i + 1) * N, 1] = radius * np.sin(theta)
        target[i * N: (i + 1) * N] = i
        plt.scatter(data[i * N: (i + 1) * N, 0], data[i * N: (i + 1) * N, 1], s=40, cmap=color)

    plt.show()
    return data, target

def split_data(attributs,result,testSize):
    attribut_train,attribut_test,result_train,result_test=train_test_split(attributs, result, test_size=testSize, random_state=4, shuffle= True)
    return attribut_train,attribut_test,result_train,result_test

def relu_derivative(output):

    output_derivative = np.zeros(output.shape)
    output_derivative[output > 0] = 1
    return output_derivative

def accuracy(data, result, probability):
    predicted = np.argmax(probability, axis=1)
    correct_prediction = np.size(result[(predicted == result)])
    print('Number of correct prediction {0}'.format(correct_prediction))
    print('correct prediction {0} %'.format((correct_prediction / float(np.size(result))) * 100))
    plot_predicted_classification(data,predicted, result)
    print('-----------------------------------------------------------------------------------------')

def plot_predicted_classification(data, predicted, result):
    K = len(set(result))
    colors = plt.cm.Spectral(np.linspace(0, 1, K))

    for i, color in zip( range(K) , colors):

        member = data[predicted == i]
        plt.scatter(member[:, 0], member[:, 1], s=40, cmap=color, marker='o')

    plt.show()

