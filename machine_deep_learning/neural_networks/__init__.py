import numpy as np
import matplotlib.pyplot as plt

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