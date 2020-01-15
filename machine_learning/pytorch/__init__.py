import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
np.random.seed(2018)


class Fashion(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        self.urls = [
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        ]
        super(Fashion, self).__init__(
            root, train=train, transform=transform, target_transform=target_transform, download=download
        )


def decode_label(l):
    return ["Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"][l]

train_data = Fashion('./data', train=True, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                            ]))

test_data = Fashion('./data', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                            ]))

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x


print(FashionModel())
train_size = train_data.train_data.shape[0]
val_size, train_size = int(0.20 * train_size), int(0.80 * train_size) # 80 / 20 train-val split
test_size = test_data.test_data.shape[0]
batch_size = 100

# Add dataset to dataloader that handles batching
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(val_size, val_size+train_size)))
val_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, val_size)))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Setup metric class
Metric = namedtuple('Metric', ['loss', 'train_error', 'val_error'])


def inference(model, loader, n_members):
    correct = 0
    for data, label in loader:
        X = data.view(-1, 784)
        Y = label
        out = model(X)
        pred = torch.argmax(out,1)
        predicted = pred.eq(Y.view_as(pred))
        correct += predicted.sum()
    return correct / n_members


class Trainer():
    """
    A simple training cradle
    """

    def __init__(self, model, optimizer, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, n_epochs):
        print("Start Training...")
        self.metrics = []
        for e in range(n_epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                self.optimizer.zero_grad()
                X = data.view(-1, 784)
                Y = label
                out = self.model(X)
                pred = torch.argmax(out,1)
                predicted = pred.eq(Y.view_as(pred))
                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            total_loss = epoch_loss/ train_size
            train_error = correct / train_size
            val_error = inference(self.model, val_loader, val_size)
            print("epoch: {0}, loss: {1:.8f}".format(e + 1, total_loss))
            self.metrics.append(Metric(loss=total_loss,
                                       train_error=train_error,
                                       val_error=val_error))

### LET'S TRAIN ###

# A function to apply "normal" distribution on the parameters

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.detach().normal_(0,1)

# We first initialize a Fashion Object and initialize the parameters "normally".
normalmodel = FashionModel()
normalmodel.apply(init_randn)

n_epochs = 8

print("SGD OPTIMIZER")
SGDOptimizer = torch.optim.SGD(normalmodel.parameters(), lr=0.01)
sgd_trainer = Trainer(normalmodel, SGDOptimizer)
sgd_trainer.run(n_epochs)
sgd_trainer.save_model('./sgd_model.pt')
print('')

