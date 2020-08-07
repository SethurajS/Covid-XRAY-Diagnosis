from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



def model_predict(IMAGE_PATH):

    img = image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    import json
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})

    import requests
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/covid:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    return predictions


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']
        result = ""
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path)

        if preds == [[1.0]]:
            result = "NORMAL"
        elif preds == [[0.0]]:
            result = "COVID"
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)