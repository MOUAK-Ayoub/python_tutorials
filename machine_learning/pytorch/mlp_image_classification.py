import torch
import torch.nn as nn
from .dataset import TrainModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = nn.Sequential(nn.Linear(32 * 32, 32), nn.ReLU(), nn.Linear(32, 10), nn.ReLU())
model_training = TrainModel(model)
model_training.train_model(2)
model_training.test_dataset()
