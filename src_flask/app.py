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
import json
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, render_template, request

# Define a flask app
app = Flask(__name__)

scoring_uri = 'http://51a0a868-51da-41ec-8c91-b7220daa47fa.eastus.azurecontainer.io/score'
scoring_uri_local = 'http://localhost:6789/score'

#Switch for testin local webservice or AZ ws
#scoring_uri = scoring_uri_local

headers = {'Content-Type': 'application/json;charset-UTF-8'}

NAME_OF_FILE = 'export.pkl' # Default convention for Fastai model, test before changing
PATH_TO_MODELS_DIR = Path() # by default just use /models in root dir
classes = ['Negative','Tumor']

def encode(img):
    img = Image.open(BytesIO(img))
    img = img.resize((500,500))
    buff = BytesIO()
    img.save(buff, format='jpeg')
    return base64.b64encode(buff.getvalue()).decode("utf-8")
	
def model_predict(img):
    img_data = preprocess(img)
    preds_raw = requests.post(scoring_uri, img_data, headers=headers)
    preds_str = preds_raw.json()
    preds_dict = json.loads(preds_str)
    pred_class = preds_dict['class']
    pred_probs = preds_dict['probs']
    
    img_data = encode(img)
       
    result = {"class":pred_class, "probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)

def preprocess(image):
    test = image
    data = str(base64.b64encode(test), encoding='utf-8')  
    input_data = json.dumps({'data': data})
    return input_data
   

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=["GET", "POST"])
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
