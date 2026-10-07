import torch
from models.alexnet import AlexNetInspired

model = AlexNetInspired(num_classes=5)

x = torch.randn(1, 3, 224, 224)

y = model(x)

print(model)

print()

print("Output Shape :", y.shape)