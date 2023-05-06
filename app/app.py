#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 14:27
# @Author  : Paulson
# @File    : app.py
# @Software: VSCode
# @define  : function

import re
import json
import base64
import numpy as np
import keras
from flask import Flask, render_template, request
from keras.utils import img_to_array, load_img

# 使用 redis 统计总访问次数，今日访问次数
# from redis_util import get_today, get_visit_num_all, get_visit_num_today, inc_visit_num

app = Flask(__name__)

model_file = './models/classifier'
global model
model = keras.models.load_model(model_file)

@app.route('/')
def index():
    # inc_visit_num()
    response = get_visit_info()
    return render_template("index.html", **response)  # 如果没有使用 redis 统计访问次数功能，请使用index.html

@app.route('/predict/', methods=['Get', 'POST'])
def preditc():
    # inc_visit_num()  # 每访问一次，增加访问次数
    parseImage(request.get_data())
    img = img_to_array(load_img('./app/output.png', target_size=(28, 28), color_mode="grayscale")) / 255.
    img = np.expand_dims(img, axis=0)
    predict_x=model.predict(img) 
    code = np.argmax(predict_x,axis=1)
    probability = predict_x[0][code]
    response = get_visit_info(int(code), float(probability))
    print(response)
    response = json.dumps(response)
    return response

def get_visit_info(code=0, probability=0):
    response = {}
    response['result'] = code
    response["probability"] = probability
    # response['visits_all'] = get_visit_num_all()
    # response['visits_today'] = get_visit_num_today()
    # response['today'] = get_today()
    return response

def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./app/output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)