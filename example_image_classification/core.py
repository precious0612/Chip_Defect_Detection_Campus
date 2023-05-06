import re
import json
import base64
import numpy as np
import keras
from flask import Flask, render_template, request
from keras.utils import img_to_array, load_img
from PIL import Image

def core(net, img_path):

    img = Image.open(img_path)
    # delta = int((img.size[0] - img.size[1]) / 2)
    # # width > height
    # if delta > 0:
    #     padding = (0, delta)
    # else:
    #     padding = (-delta, 0)

    img = img_to_array(load_img(img_path, target_size=(32, 32), color_mode="rgb")) / 255.
    img = np.expand_dims(img, axis=0)
    prob=net.predict(img)[0]
    print(prob)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ouptut_prob = {}
    for i, value in enumerate(prob):
        ouptut_prob[classes[i]] = "{:.2f}%".format(float(value) * 100)
    return {
        "max_class": classes[np.argmax(prob)],
        "probability": ouptut_prob
    }

if __name__ == "__main__":
    model_file = './models/classifier'
    net = keras.models.load_model(model_file)
    result_data = core(net, "/Users/precious/BackGround/b.jpg")
    print(result_data)

