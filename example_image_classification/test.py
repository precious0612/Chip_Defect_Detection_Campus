import webbrowser
import json
import keras
import base64
import os, sys

from flask import Flask, request, jsonify, json, render_template
from core import core


app = Flask(__name__)
model_file = './models/classifier'
global net
net = keras.models.load_model(model_file)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start', methods=['post'])
def start():
    data = json.loads(request.get_data())
    img = base64.b64decode(data["img"])
    img_path = "./example_image_classification/static/images/example.jpg"

    with open(img_path, 'wb') as file:
        file.write(img)

    result_data = core(net, img_path)
    return json.dumps(result_data)

if __name__ == '__main__':

    # webbrowser.open("http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=8000, debug=True)

