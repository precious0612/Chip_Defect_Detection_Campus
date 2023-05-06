#获取CIFAR10数据集
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

train_data = datasets.CIFAR10(root="data/",train=True,transform=transforms.ToTensor(),
                              download=True)
test_data = datasets.CIFAR10(root="data/",train=False,transform=transforms.ToTensor(),
                             download=False)

train_laoder = DataLoader(train_data,batch_size=100,shuffle=True)
for i,(img,label) in enumerate(train_laoder):
    print(i)
    print(img.shape)
    print(label.shape)
    print(img[0])
    exit()


# print(train_data)
# print(test_data)
#
# print(train_data.data.shape)
# print(train_data.targets)
# print(train_data.classes)
#
# img_data = train_data.data[5]
# label_index = train_data.targets[5]
# label = train_data.classes[label_index]
# print(label)
# print(img_data)
# unloader = transforms.ToPILImage()
# img = unloader(img_data)
# img.show()