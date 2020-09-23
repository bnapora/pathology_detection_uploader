import os, base64
import glob
import re
import torch
import json
from io import BytesIO
from azureml.core.model import Model
from azureml.core import Workspace
import fastai 
from fastai.vision.all import *
from fastai.callback.all import *
import urllib.request



def init():
    global learner
    pathsrc=Path()

    model_path=os.getenv('AZUREML_MODEL_DIR')  
    
    filename="export.pkl"
    classes = ['Positive','Negative']
    model_path_file = os.path.join(os.environ['AZUREML_MODEL_DIR'], filename)
    model_path_hc = 'azureml-models/breastcancerfastai/1'
    print('AZUREML_MODEL_DIR-Model_Path: ' + model_path)
    print('Model_Path_File: ' + model_path_file)
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
    classes = learner.dls.vocab
    print(learner.dls.vocab)

def encode(img):
    img = Image.open(BytesIO(img))
    img = img.resize((500,500))
    buff = BytesIO()
    img.save(buff, format='jpeg')
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def preprocess(image):
    test = image
    data = str(base64.b64encode(test), encoding='utf-8')  
    input_data = json.dumps({'data': data})
    return input_data

def run(raw_data):
    base64_string = json.loads(raw_data)['data']
    base64_bytes = base64.b64decode(base64_string)

    # img = Image.open(BytesIO(base64_bytes))
    img = PILImage.create(BytesIO(base64_bytes))

    pred_class,pred_idx,outputs = learner.predict(img)

    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learner.dls.vocab, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )

    #img_data = preprocess(img)

    result = {'class':pred_class, "probs":pred_probs}
    # result = {"class":pred_class, "probs":pred_probs, "image":img_data}
    # return json.dumps({'class':str(result[0]), 'probs':result[2].data[1].item()})
    return json.dumps(result)