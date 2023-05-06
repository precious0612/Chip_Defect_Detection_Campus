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
            nn.Conv2d(1,32,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
    def forward(self,x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1)
        )

    def forward(self,x):
        return self.layers(x)

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