import torch
from torch import nn
import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Encoder(nn.Module):
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

    def forward(self,x):
        x = x.reshape(-1,28*28)
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1*28*28)
        )

    def forward(self,x):
        return self.layers(x).reshape(-1,1,28,28)

class Main_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        encoder_out = self.encoder(x)
        out = self.decoder(encoder_out)
        return out

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    #判断目录并创建
    if not os.path.exists("img"):
        os.makedirs("img")

    train_data = datasets.MNIST("data/",train=True,transform=transforms.ToTensor(),download=True)
    train_laoder = DataLoader(train_data,batch_size=100,shuffle=True)

    net = Main_Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    k = 0
    for epoch in range(10000):
        for i,(img,_) in enumerate(train_laoder):
            img = img.to(DEVICE)
            out = net(img)
            loss = loss_func(out,img)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%10==0:
                print(loss.item())
                fack_img = out.detach()
                save_image(fack_img,"img/{}-fack_img.png".format(k),nrow=10)
                save_image(img, "img/{}-real_img.png".format(k), nrow=10)
                k=k+1