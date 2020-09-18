from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests
import base64

# Import fast.ai Library
import torch
from fastai import *
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.vision.core import *


# Flask utils
from flask import Flask, redirect, url_for, render_template, request
#from PIL import Image as pilImage

# Define a flask app
app = Flask(__name__)

NAME_OF_FILE = 'export.pkl' # Name of your exported file
PATH_TO_MODELS_DIR = Path() # by default just use /models in root dir
classes = ['Negative','Tumor']

# Include functions from model class
def get_x(r): 
    pathstr = str(path)
    idstr = str(r['id'])
    imgpathstr = pathstr + '/train/' + idstr + '.tif'
    return imgpathstr
def get_y(r): return r['label']

def change_pred_totext(preds):
    if (preds==1):
        predvalue = 'Positive'
    else:
        predvalue = 'Negative'
    return predvalue

def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    learn = load_learner(path_to_pth_file/'export.pkl')
    return learn

learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

def encode(img):
    img = Image.open(BytesIO(img))
    img = img.resize((500,500))
    buff = BytesIO()
    img.save(buff, format='jpeg')
    return base64.b64encode(buff.getvalue()).decode("utf-8")
	
def model_predict(img):
    #img = Image.open(BytesIO(img))
    pred_class,pred_idx,outputs = learn.predict(img)
    pred_class = change_pred_totext(pred_class)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learn.dls.vocab, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )
   
    img_data = encode(img)
    result = "Testing"
    result = {"class":pred_class, "probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)
   

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        if img != None:
        # Make prediction
            preds = model_predict(img)
            return preds
    return 'OK'
	
@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content)
            return preds
    return 'OK'
    

if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        app.run(debug=True, host='0.0.0.0', port=port)
