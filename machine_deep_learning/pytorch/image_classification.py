import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, trainloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    criterion = nn.CrossEntropyLoss()
    loss_array = []

    for batchid, (image, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_array.append(loss)
    return np.min(loss_array)

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.flatten(x, 1)),
     transforms.Lambda(lambda x: torch.squeeze(x)),
     transforms.Lambda(lambda x: x.to(device))
     ])

transform_target = transforms.Lambda(lambda x: torch.tensor(x).to(device))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, target_transform=transform_target )
trainloader  = torch.utils.data.DataLoader(trainset, batch_size=50)

model = nn.Sequential(nn.Linear(32 * 32, 32), nn.ReLU(), nn.Linear(32, 10), nn.Softmax())
model = model.to(device)
loss_epochs= []

for i in range(num_epochs):
    curr =time.time()
    loss = train_model(model,trainloader)
    loss_epochs.append(loss)
    print("loss for epoch {0} is {1}".format(i, loss))
    print("the {0} epoch took {1}".format(i, str(datetime.timedelta(seconds=time.time()-curr))))


plt.plot(range(num_epochs), loss_epochs)
plt.show()

