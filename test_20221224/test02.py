import torch
from torch import nn

class Net_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.out_layer = nn.Linear(512*2*2,10)
    def forward(self,x):
        conv_out =  self.conv_layers(x)
        conv_out = conv_out.reshape(-1,512*2*2)
        return self.out_layer(conv_out)

if __name__ == '__main__':
    net = Net_v1()
    x = torch.randn(1,3,32,32)
    y = net.forward(x)
    print(y.shape)