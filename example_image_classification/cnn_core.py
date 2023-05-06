import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def core(net, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.eval()

    img = Image.open(img_path)
    # delta = int((img.size[0] - img.size[1]) / 2)
    # # width > height
    # if delta > 0:
    #     padding = (0, delta)
    # else:
    #     padding = (-delta, 0)

    transform = transforms.Compose([transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img = transform(img).unsqueeze(0)
    output = net(img.to(device))

    # generate probability
    prob = F.softmax(output, dim=1)[0]
    print(prob)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ouptut_prob = {}
    for i, value in enumerate(prob):
        ouptut_prob[classes[i]] = str(round(value.data.item() * 100, 2)) + "%"
    return {
        "max_class": classes[torch.argmax(output).cpu().item()],
        "probability": ouptut_prob
    }

if __name__ == "__main__":
    net = torch.load("models/net.pth")
    result_data = core(net, "images/bird.jpg")
    print(result_data)

