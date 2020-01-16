import torch
import torch.nn as nn
from machine_learning.pytorch.dataset import TrainModel



model_adam = nn.Sequential(nn.Linear(32 * 32, 32), nn.ReLU(), nn.Linear(32, 10))
adam_optim = torch.optim.Adagrad(model_adam.parameters())
model_training = TrainModel(model_adam, adam_optim)
model_training.train_model(10)
model_training.test_dataset()