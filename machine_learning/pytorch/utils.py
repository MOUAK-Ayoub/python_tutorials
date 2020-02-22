import torch.nn as nn
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from . import nn_classifier
from . import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_input = 32*32
data_output = 10
hidden_layer = [124]
model_default = models.MLP(data_input, data_output, hidden_layer)

activations = [nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.Sigmoid(), nn.Threshold(1, 1e-4)]
transform_linear = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x, 1)),
    transforms.Lambda(lambda x: torch.squeeze(x)),
    transforms.Lambda(lambda x: x.to(device))
])
transform_cnn = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device))
])


def get_tranform(model):
    transform = transform_cnn
    for id, child in enumerate(model.children()):

        if id > 1:
            break
        if isinstance(child, nn.Linear):
            transform = transform_linear
            break
        if isinstance(child, nn.Sequential):
            transform = get_tranform(child)
            break
    return transform


def is_loss_decreasing(loss_array, threshold=0.05, elements=4):
    if len(loss_array) < elements:
        return
    diff = np.diff(loss_array[-elements:])
    loss_increase = np.sum(diff > threshold) == elements - 1
    if loss_increase:
        print('Loss is increasing ')
        return
    loss_decrease = np.sum(np.absolute(diff) < threshold) == elements - 1
    print('Loss is {0} decreasing significantly by threshold of {1} '
          .format('not' if loss_decrease else '', threshold))


def models_by_dropouts(min=0, max=0.9, length=2):
    dropouts = np.linspace(min, max, length)
    model_array = []
    for dropout in dropouts:
        model_sample = models.MLP(data_input, data_output, hidden_layer, p=dropout)
        model_array.append((model_sample, None))
    return model_array


def models_by_lr(start=1e-3, stop=1, length=10):
    model_mlp = models.MLP(data_input, data_output, hidden_layer)
    model_array = []
    lrs = np.linspace(start, stop, length)
    for lr in lrs:
        optimizer = torch.optim.SGD(model_mlp.parameters(), lr=lr)
        model_sample = model_mlp, optimizer
        model_array.append(model_sample)
    return model_array


def models_by_hlayerlength(start=1, stop=8):
    model_array = []
    length_hlayer_array = 2 ** np.arange(start, stop)
    for n in length_hlayer_array:
        model = nn.Sequential(nn.Linear(32 * 32, n), nn.ReLU(), nn.Linear(n, 10))
        model_array.append((model, None))
    return model_array


def models_by_activation():
    model_array = []
    for activation in activations:
        model = models.MLP(data_input, data_output, hidden_layer, activ=activation)
        model_array.append((model, None))
    return model_array


def models_by_depth(number):
    assert(number > 1)
    model_array = []
    for i in range(1, number):
        model = model_depth(i)
        model_array.append((model, None))
    return model_array


def model_depth(n):
    if n >= np.log2(data_input) or n <= 0:
        print('Hidden layers must be in the interval ]0,{0}['.format(np.log2(data_input)))
        return
    hidden_layers = np.array(data_input / (2 ** np.arange(1, n+1)), dtype=int)
    model = models.MLP(data_input, data_output, hidden_layers)

    return model


def models_by_optim():
    optimizers = [torch.optim.SGD(model_default.parameters(), lr=1e-2),
                  torch.optim.SGD(model_default.parameters(), lr=1e-2, momentum=0.9),
                  torch.optim.SGD(model_default.parameters(), lr=1e-2, momentum=0.5),
                  torch.optim.SGD(model_default.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
                  torch.optim.Adam(model_default.parameters()),
                  torch.optim.RMSprop(model_default.parameters())]
    models_array = []
    for optimizer in optimizers:
        models_array.append((model_default, optimizer))
    return models_array


def plot_accuracy_by_criteria(model_array, epochs=10):
    accuracy = []
    loss = []
    for model_id, (model, optimizer) in enumerate(model_array):
        model_training = nn_classifier.TrainModel(model, optimizer)
        model_training.train_model(epochs)
        accuracy_test, loss_test = model_training.test_model()
        accuracy.append(accuracy_test)
        loss.append(loss_test)
        if model_id % 5 == 0:
            print('The model {0} had an accuracy of {1}'.format(model_id, accuracy_test))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.plot(range(model_id+1), accuracy, label='Accuracy ', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(range(model_id+1), loss, label='Loss ', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper left')
    fig.tight_layout()
    plt.show()
    print('The best model for max accuracy is model N° {0} with an accuracy of {1} '
          .format(np.argmax(accuracy), np.max(accuracy)))
    print('The best model  for min loss is model N° {}'.format(np.argmin(loss)))


