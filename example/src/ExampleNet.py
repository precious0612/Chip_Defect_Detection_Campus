import torch
from torch import nn

class ExampleNet(nn.Module):
    #模型设计
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1*28*28,512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    #模型的使用（前向计算）
    def forward(self,x):
        #数据变形：NCHW==》NV
        x = x.reshape(-1,1*28*28)
        return self.layers(x)
if __name__ == '__main__':
    #创建网络对象
    net = ExampleNet()
    #创建虚拟数据
    x = torch.randn(3,1,28,28)
    #使用模型得到输出
    y = net.forward(x)
    print(y.shape)