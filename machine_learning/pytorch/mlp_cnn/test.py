
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from machine_learning.pytorch.mlp_cnn import nn_classifier


class CNN3Filter(nn.Module):

    def __init__(self):
        super(CNN3Filter, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 3)
        self.conv2 = nn.Conv2d(2, 10, 18)
        self.maxpool = nn.MaxPool2d(13)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.maxpool(x)
        return torch.flatten(out, 1)
model = CNN3Filter()
model_training = nn_classifier.TrainModel(model)
model_training.train_model(1)
model_training.test_model()
weights = torchvision.utils.make_grid(model.conv1.weight)
img_np = weights.detach().numpy()
plt.imshow(np.transpose(img_np, (1, 2, 0)))
plt.show()


