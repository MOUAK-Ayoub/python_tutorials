import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
from  machine_learning.pytorch.dataset import TrainModel


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.mlp = nn.Sequential(nn.Linear(32 * 32, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        out = self.mlp(x)
        return out


model = cnn()
model_training = TrainModel(model)
model_training.train_model(1)
model_training.test_dataset()


















# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.to(device))
# ])
# transform_target = transforms.Lambda(lambda x: torch.tensor(x).to(device))

# dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform,
#                                        target_transform=transform_target)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
#

#
# def show(img):
#
#     image = torch.squeeze(img)
#     image = transforms.ToPILImage()(image)
#     plt.imshow(image)
#
# filter =  torch.tensor([[0,-1,0],
#                         [-1,4,-1],
#                         [0,-1,0]]
#                        ).float()
# filter = filter.view(1,1,3,3).repeat(1,3,1,1)
#
# for batchid, (images, target) in enumerate(dataloader):
#
#     if batchid > 3:
#         break
#
#     image_conv = func.conv2d(images,filter, padding=1)
#
#     for i in range(images.size(0)):
#         show(images[i])
#         plt.show()
#         show(image_conv[i])
#         plt.show()
