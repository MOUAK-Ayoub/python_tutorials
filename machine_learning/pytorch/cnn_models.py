import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.mlp = nn.Sequential(nn.Linear(16 * 16, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        out = self.mlp(x)
        return out


class CNNGray(nn.Module):

    def __init__(self, input_size, output_size, filter_size):
        super(CNNGray, self).__init__()
        self.conv = nn.Conv2d(1, output_size, filter_size)
        self.maxpool = nn.MaxPool2d(input_size-filter_size+1)

    def forward(self, x):
        x = self.conv(x)
        out = self.maxpool(x)
        return torch.flatten(out, 1)


class CNNRgb(nn.Module):

    def __init__(self, input_size, output_size, pattern_size):
        super(CNNRgb, self).__init__()
        self.conv = nn.Conv2d(3, output_size, pattern_size)
        self.maxpool = nn.MaxPool2d(input_size-pattern_size+1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        out = self.maxpool(x)
        return torch.flatten(out, 1)


class CNN2Filter(nn.Module):

    def __init__(self, input_size, output_size, pattern_size, filter_size, num_features):
        super(CNN2Filter, self).__init__()
        self.conv1 = nn.Conv2d(3, num_features, filter_size)
        self.conv2 = nn.Conv2d(num_features, output_size, pattern_size-filter_size+1)
        self.maxpool = nn.MaxPool2d(input_size-pattern_size+1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.maxpool(x)
        return torch.flatten(out, 1)


class CNN3Filter(nn.Module):

    def __init__(self):
        super(CNN3Filter, self).__init__()
        self.conv1 = nn.Conv2d(3, 40, 4)
        self.conv2 = nn.Conv2d(40, 30, 8)
        self.conv3 = nn.Conv2d(30, 10, 10)
        self.maxpool = nn.MaxPool2d(13)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.maxpool(x)
        return torch.flatten(out, 1)


class CNNMultiFilterWithMaxPool(nn.Module):

    def __init__(self, input_size, output_size, filter_array, num_features):
        super(CNNMultiFilterWithMaxPool, self).__init__()
        self.conv1 = nn.Conv2d(3, num_features, filter_array[0])
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(num_features, output_size, filter_array[1])
        self.maxpool2 = nn.MaxPool2d(int((input_size-filter_array[0]+1)/2 - filter_array[1]+1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        out = self.maxpool2(x)
        return torch.flatten(out, 1)
















