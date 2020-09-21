import os, base64
import torch
import json
from azureml.core.model import Model
from azureml.core import Workspace
import fastai 
from fastai.vision import *
from fastai.callback.all import *
from fastai.metrics import accuracy 
from fastai.metrics import error_rate
import urllib.request

    # global model

    # # The AZUREML_MODEL_DIR environment variable indicates
    # # a directory containing the model file you registered.
    # model_filename = 'sklearn_regression_model.pkl'
    # model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    # model = joblib.load(model_path)

def init():   
    global learner
    # The AZUREML_MODEL_DIR environment variable indicates a directory containing the model file you registered.  
    #this init works 
    model_path=os.getenv('AZUREML_MODEL_DIR')     
    filename="export.pkl"
    classes = ['Positive','Negative']
    learner = load_learner(path=model_path, file=filename)   
    classes = learner.data.classes
    print(classes)

def run(raw_data):
    base64_string = json.loads(raw_data)['data']
    base64_bytes = base64.b64decode(base64_string)
    with open(os.path.join(os.getcwd(),"score.jpg"), 'wb') as f:
        f.write(base64_bytes)
    
    # make prediction
    #img = open_image(os.path.join(os.getcwd(),"score.jpg"))
    #result = learn.predict(img)
    result = 'Test,Positive'
    # return json.dumps({'class':str(result[0]), 'probs':result[2].data[1].item()})
    return json.dumps({'class':str(result[0])})