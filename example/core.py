import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def core(net, img_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.eval()

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    img = transform(Image.open(img_path)).unsqueeze(0) * -1
    output = net(img.to(device))

    # generate probability
    ouptut_prob = {}
    for i, value in enumerate(F.softmax(output, dim=1)[0]):
        ouptut_prob[i] = str(round(value.data.item() * 100, 2)) + "%"
    return {
        "max_number": str(torch.argmax(output).cpu().numpy()),
        "probability": ouptut_prob
    }


