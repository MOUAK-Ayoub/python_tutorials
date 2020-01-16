import torch.nn as nn
import torchvision.transforms as transforms
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        break;
    return transform
