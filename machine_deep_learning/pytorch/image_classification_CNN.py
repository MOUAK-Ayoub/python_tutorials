
import torch.nn as nn
import machine_deep_learning.pytorch.mlp_image_classification as mlp

print('test')

model = nn.Sequential(nn.Conv2d(1,10,3),
                      nn.ReLU(),
                      nn.Conv2d(10,1,3),
                      nn.Flatten(),
                      nn.Linear(32 * 32, 10),
                      nn.ReLU())
model = model.to(mlp.device)
mlp.train_model(model,mlp.init_dataset())