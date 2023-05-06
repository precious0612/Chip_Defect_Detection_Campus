
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from src.ExampleNet import ExampleNet
# from src.ResNet import net
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	epochs = 120

	batch_size = 100

	net = ExampleNet().to(device)
	# net.to(device)
	# net = torch.load(os.path.join(sys.path[0], "models/net.pth"),map_location=device)

	criterion = nn.CrossEntropyLoss(reduce = None, weight = None, size_average = None, ignore_index = -100)

	optimizer = optim.Adam(net.parameters(), weight_decay = 0, amsgrad = False, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08)

	transform = transforms.Compose([
	    transforms.Resize(32),
	    transforms.CenterCrop(32),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	dataset = datasets.CIFAR10("datasets/", train=True, download=True,transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)

	test_transform = transforms.Compose([
	    transforms.Resize(32),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	testdataset = datasets.CIFAR10("datasets/", train=False, download=True,transform=test_transform)
	testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

	losses = []
	for i in range(epochs):

		for j, (input, target) in enumerate(dataloader):
			input, target = input.to(device), target.to(device)
			output = net(input)
			loss = criterion(output, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if j % 10 == 0:
				losses.append(loss.cpu().detach().numpy())
				print("[epochs - {0} - {1}/{2}]loss: {3}".format(i, j, len(dataloader), loss.float()))
				plt.clf()
				plt.plot(losses)
				plt.pause(0.01)
			with torch.no_grad():
				net.eval()
				correct = 0.
				total = 0.
				for input, target in testdataloader:
					input, target = input.to(device), target.to(device)
					output = net(input)
					_, predicted = torch.max(output.data, 1)
					total += target.size(0)
					correct += (predicted == target).sum()
					accuracy = correct.float() / total
				net.train()
				print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
		torch.save(net, "models/net.pth")


