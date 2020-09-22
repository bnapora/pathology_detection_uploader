import os, base64
import torch
import json
from azureml.core.model import Model
from azureml.core import Workspace
import fastai 
# from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
# from fastai.metrics import accuracy 
# from fastai.metrics import error_rate
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
    pathsrc=Path()

    model_path=os.getenv('AZUREML_MODEL_DIR')  
    
    filename="export.pkl"
    classes = ['Positive','Negative']
    model_path_file = os.path.join(os.environ['AZUREML_MODEL_DIR'], filename)
    model_path_hc = 'azureml-models/breastcancerfastai/1'
    model_path_file_hc = './source_directory/azureml-models/breastcancerfastai/1/export.pkl'
    print('AZUREML_MODEL_DIR-Model_Path: ' + model_path)
    print('Model_Path_File: ' + model_path_file)
    print('Model_Path_File_HC: ' + model_path_file_hc)
    print('FastAI Version: ' + fastai.__version__)

    print ('Path(): ' + str(pathsrc))

    for filename in os.listdir(pathsrc):
        print(filename)

    for dirname, dirnames, filenames in os.walk(pathsrc):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            print(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            print(os.path.join(dirname, filename))

    learner = load_learner(model_path_file)
    # classes = learner.data.classes
    # print('Learner: ' + classes)


def run(raw_data):
    # base64_string = json.loads(raw_data)['data']
    # base64_bytes = base64.b64decode(base64_string)
    # with open(os.path.join(os.getcwd(),"score.jpg"), 'wb') as f:
    #     f.write(base64_bytes)
    
    # make prediction
    #img = open_image(os.path.join(os.getcwd(),"score.jpg"))
    #result = learn.predict(img)
    result = 'Test,Positive'
    # return json.dumps({'class':str(result[0]), 'probs':result[2].data[1].item()})
    return result