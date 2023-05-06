import os
import io
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.autograd import Variable
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms#, utils
import numpy as np
from PIL import Image
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from flask import Flask, jsonify, request, render_template, send_file
from crop import label_mask
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_img_path = "./WaterWatch/test_data/test_images/example.jpg"
test_mask_path = "./WaterWatch/test_data/u2netp_results/example.png"
# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("\\")[-1]
    # print(image_name)
    # print(img_name)
    image = cv2.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    # print(aaa)
    # print(bbb)
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    # imo.save(d_dir+imidx+'.png')
    return imo

def get_prediction(image_bytes):
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    image = Image.open(io.BytesIO(image_bytes))
    image.save("./WaterWatch/test_data/test_images/example.jpg")

    image_dir = './data/intel/test/test_images/'
    prediction_dir = './data/intel/test/' + model_name + '_results/'
    model_dir = "/Users/precious/defect_detecting/models/u2net/u2netu2net_bce_itr_13_train_0.355659_tar_0.049504.pth"

    img_name_list = glob.glob(image_dir + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir,map_location=device))
    # net = torch.load(model_dir,map_location=device)
    if torch.cuda.is_available():
        net.cuda()
    #开启测试模式
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        return save_output(img_name_list[i_test],pred,prediction_dir)

@app.route('/')
def index():
    return render_template("client.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        get_prediction(image_bytes=img_bytes).save(test_mask_path)
        label_mask(test_img_path,test_mask_path)
        return send_file('static/example.png', mimetype='image/gif')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000,debug=True)