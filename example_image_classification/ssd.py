# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:53:54 2022
trainval.py
"""

import torch
import torchvision
import torchvision.transforms as transforms

import argparse, os
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18

transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = torchvision.datasets.CIFAR10(root='datasets/', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='datasets/', train=False,
                                       download=True, transform=transform)


def train(args, net, device, train_loader, optimizer, epoch, scheduler):
    running_loss = 0.0  # 初始化loss
    correct = 0.

    batch_num = 0
    # 重点注意，训练时如果用到Batch Normalization 和 Dropout，就要在训练时使用net.train(),测试时用net.eval(),否则则不用
    net.train()
    criterion = nn.CrossEntropyLoss()  # nn的函数是要先创建，后初始化

    # 开始数据机加载batch
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        # 输入数据上传

        inputs = inputs.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        batch_num += 1

        pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # 更新参数
        optimizer.step()

        if batch_idx % args.log_interval == 0:  # 每args.log_interval个批次输出一次loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),
                       100 * (batch_idx + 1) * len(inputs) / len(train_loader.dataset), running_loss / (batch_idx + 1)))
    scheduler.step()


def test(args, net, device, test_loader, train_loader, epoch):
    net.eval()  # 用到Batch Normalization 和 Dropout 就要加上
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # nn的函数是要先创建，后初始化
    with torch.no_grad():
        for data, label in test_loader:  # 不会做反向求导
            data, label = data.to(device), label.to(device)
            output = net(data)

            test_loss += F.cross_entropy(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    # 获取参数
    args = parser.parse_args()

    # 先来判断是否要用cuda,默认是有的话就用

    torch.manual_seed(args.seed)  # 阈值随机设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 准备数据加载器
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # 初始化net,训练和验证都需要net
    net = resnet18(pretrained=True)
    #net.load_state_dict(torch.load("resnet18-5c106cde.pth"))  # 加载官方预训练模型，6个epoch 95.26%

    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 10)

    net = net.to(device)

    print("Create Net:", net)

    # 初始化optimizer，只有train时使用
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)

    # 开始迭代训练
    for epoch in range(args.epochs):
        train(args, net, device, train_loader, optimizer, epoch, scheduler)
        test(args, net, device, test_loader, train_loader, epoch)  # 不需要optimizer
        if (args.save_model):
            torch.save(net, "models/cnn_resnet18.pth")  # 不使用state_dict()，则将模型结构和权重一起保存


if __name__ == "__main__":
    main()