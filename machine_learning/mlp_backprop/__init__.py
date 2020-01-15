import torchvision
import torchvision.transforms as transforms
import torch


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms)

print(type(trainset))
trainloader  = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
