import torch.nn as nn
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from . import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_input = 32*32
data_output= 10
model_default = nn.Sequential(nn.Linear(32 * 32, 256), nn.ReLU(), nn.Linear(256, 10))
optimizers = [torch.optim.SGD(model_default.parameters(), lr=1e-2),
              torch.optim.SGD(model_default.parameters(), lr=1e-2, momentum=0.9),
              torch.optim.SGD(model_default.parameters(), lr=1e-2, momentum=0.5),
              torch.optim.SGD(model_default.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
              torch.optim.Adam(model_default.parameters()),
              torch.optim.RMSprop(model_default.parameters())]
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
    for child in model.children():
        if isinstance(child, nn.Linear):
            transform = transform_linear
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
    print(' Loss is {0} decreasing significantly by threshold of {1} '
          .format('not' if loss_decrease else '', threshold))


def models_by_lr(start=1e-3 , stop=1, length=10):
    model = nn.Sequential(nn.Linear(32 * 32, 256), nn.ReLU(), nn.Linear(256, 10))
    models = []
    lrs = np.linspace(start, stop, length)
    for lr in lrs:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model_sample = model, optimizer
        models.append(model_sample)
    return models


def models_by_hlayerlength(start=1, stop=8):
    models = []
    length_hlayer_array = 2 ** np.arange(start, stop)
    for n in length_hlayer_array:
        model = nn.Sequential(nn.Linear(32 * 32, n), nn.ReLU(), nn.Linear(n, 10))
        models.append((model, None))
    return models


def models_by_activation(depth=1):
    models = []
    for activation in activations:
        model = model_depth(depth, activation)
        models.append((model, None))
    return models


def models_by_depth(number):
    models = []
    for i in range(number):
        model = model_depth(i)
        models.append(model)
    return models


def model_depth(n, activ=nn.ReLU()):
    if n >= np.log2(data_input) or n <= 0 :
        print('Hidden layers must be in the interval ]0,{0}['.format(np.log2(32 * 32)))
        return

    hidden_length = np.array(data_input /(2 ** np.arange(0, n+1)), dtype=int)
    layers = []
    for i in range(n):
        layers.append(nn.Linear(hidden_length[i], hidden_length[i+1]))
        layers.append(activ)
    layers.append(nn.Linear(hidden_length[n], data_output))
    model = nn.Sequential(*layers)
    return model


def models_by_optim():
    models = []
    for optimizer in optimizers:
        models.append((model_default, optimizer))
    return models


def plot_accuracy_by_criteria(models, epochs=10):
    accuracy = []
    loss = []
    for model_id, (model, optimizer) in enumerate(models):
        model_training = dataset.TrainModel(model, optimizer)
        model_training.train_model(epochs)
        accuracy_test, loss_test = model_training.test_dataset()
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
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(range(model_id+1), loss, label='Loss ', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper left')
    fig.tight_layout()
    plt.show()
    print('The best model for max accuracy is model N° {0} with an accuracy of {1} '.format(np.argmax(accuracy), np.max(accuracy) ))
    print('The best model  for min loss is model N° {}'.format(np.argmin(loss)))


