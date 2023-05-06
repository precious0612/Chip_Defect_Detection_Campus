from torchvision import models
from torch import nn

net = models.resnet18(pretrained=True)
net.fc = nn.Linear(512,10)