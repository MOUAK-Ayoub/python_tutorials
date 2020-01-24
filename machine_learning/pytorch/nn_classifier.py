import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from . import utils
import matplotlib.pyplot as plt
import time, datetime
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainModel:

    def __init__(self, model,
                 optimizer=None,
                 loss_criterion=nn.CrossEntropyLoss(),
                 train=True):

        self.model = model.to(device)
        if optimizer == None:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        else:
            self.optimizer = optimizer
            self.optimizer.params = model.parameters()
        self.loss_criterion = loss_criterion
        self.dataloader = self.init_dataset(train)

    def init_dataset(self, train):

        transform = utils.get_tranform(self.model)
        transform_target = transforms.Lambda(lambda x: torch.tensor(x).to(device))

        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, transform=transform, download=True,
                                                target_transform=transform_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
        return dataloader, len(dataset)

    def train_model(self, num_epoch):
        loss_epochs = []

        for i in range(num_epoch):
            curr = time.time()
            loss = self.train_epoch()
            loss_epochs.append(loss.item())
            if num_epoch % 10 == 0:
                print("loss for epoch {0} is {1}".format(i, loss))
                end = time.time()
                print("the {0} epoch took {1}".format(i, str(datetime.timedelta(seconds=end - curr))))

        plt.plot(range(num_epoch), loss_epochs)
        plt.show()
        utils.is_loss_decreasing(loss_epochs)

    def train_epoch(self):
        dataloader, size = self.dataloader
        optimizer = self.optimizer
        criterion = self.loss_criterion
        loss_array = []
        accuracy = 0
        for batchid, (image, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = self.model(image)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            loss_array.append(loss.item())
            accuracy += torch.sum(torch.argmax(output, 1) == target)

        # accuracy per epoch
        # print('accuracy {}%'.format(accuracy/float(size)*100))
        return np.mean(loss_array)

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            accuracy = 0
            loss_array = []
            testloader, size = self.init_dataset(False)
            for batchid, (image, target) in enumerate(testloader):
                output = self.model(image)
                loss = self.loss_criterion(output, target)
                loss_array.append(loss.item())
                accuracy += torch.sum(torch.argmax(output, 1) == target)

        print('Test mean loss : {}'.format(np.mean(loss_array)))
        print('Test accuracy: {}%'.format(accuracy/float(size)*100))
        return accuracy/float(size)*100, np.mean(loss_array)




