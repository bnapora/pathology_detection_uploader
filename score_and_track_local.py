import json
import torch
import os, base64
from fastai import *
from fastai.vision import *
from azureml.core.model import Model

from azureml.core import Workspace

#from azureml.monitoring import ModelDataCollector
def initazws():
    global ws
    subscription_id = "40bffbcc-578f-4e44-bd6d-972552eb6513" # The ID of the Azure Subscription
    resource_group = "gestaltml" # Name of a logical resource group
    workspace_name = "fastai2" # The name of the workspace to look for or to create
    workspace_region = 'East US' # Location of the workspace
    #experiment_name = 'breastcancer'
    score_script = 'score_and_track.py'
    modelupload_name = 'breastcancerdetect'
    service_name = 'breastcancerdetect'

    ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
def init():
    global learn
    
    #path=os.getenv('AZUREML_MODEL_DIR')

    model_file = Model.get_model_path('breastcancerdetect')
    model_path = os.path.dirname(model_file)

    learn = load_learner(model_path)
    
    
def run(raw_data):
    base64_string = json.loads(raw_data)['data']
    base64_bytes = base64.b64decode(base64_string)
    with open(os.path.join(os.getcwd(),"score.jpg"), 'wb') as f:
        f.write(base64_bytes)
    
    # make prediction
    img = open_image(os.path.join(os.getcwd(),"score.jpg"))
    #result = learn.predict(img)
    result = 'Test,1'
    # return json.dumps({'class':str(result[0]), 'probs':result[2].data[1].item()})
    return json.dumps({'class':str(result[0])})