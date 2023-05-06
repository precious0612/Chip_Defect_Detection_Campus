import webbrowser
import json
import torch
import base64
import os, sys

from flask import Flask, request, jsonify, json, render_template
from core import core


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(os.path.join(sys.path[0],"models/net.pth"),map_location=device)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/start', methods=['post'])
def start():
	data = json.loads(request.get_data())
	img = base64.b64decode(data["img"])
	img_path = "static/images/example.jpg"

	with open(img_path, 'wb') as file:
		file.write(img)

	result_data = core(net, img_path)
	return json.dumps(result_data)

if __name__ == '__main__':

	# webbrowser.open("http://127.0.0.1:5000")
	app.run(host='127.0.0.1')

