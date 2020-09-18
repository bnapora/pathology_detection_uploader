import json
import torch
import os, base64
from fastai import *
from fastai.vision import *
from azureml.core.model import Model
#from azureml.monitoring import ModelDataCollector

def init():
    global learn
    
    model_file = Model.get_model_path('breastcancerdetect_fastai2')
    model_path = os.path.dirname(model_file)

    #learn = load_learner(model_path)
    
    
def run(raw_data):
    base64_string = json.loads(raw_data)['data']
    base64_bytes = base64.b64decode(base64_string)
    with open(os.path.join(os.getcwd(),"score.jpg"), 'wb') as f:
        f.write(base64_bytes)
    
    # make prediction
    img = open_image(os.path.join(os.getcwd(),"score.jpg"))
    #result = learn.predict(img)
    result = 'Test,1'
    return json.dumps({'class':str(result[0]), 'probs':result[2].data[1].item()})