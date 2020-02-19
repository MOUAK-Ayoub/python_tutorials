import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_layer, activ=nn.ReLU(), p=0): # No dropout by default
        super(MLP, self).__init__()
        assert(len(hidden_layer) > 0)
        layers = [nn.Dropout(p), nn.Linear(input_shape, hidden_layer[0])]
        for i in range(len(hidden_layer[:-1])):
            layers.append(nn.Dropout(p))
            layers.append(nn.Linear(hidden_layer[i], hidden_layer[i + 1]))
            layers.append(activ)
        layers.append(nn.Linear(hidden_layer[-1], output_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x






