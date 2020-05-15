from PIL import Image
import base64
import io
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize
from tensorflow import reshape
from azureml.core.model import Model

def init():
    global model, img_size

    print("Working directory")
    print(os.getcwd())
    for root, dirs, files in os.walk('.', topdown=False):
        print(root)
        print(dirs)
        print(files)

    with open('./inference/model_tags.json', 'r') as file:
        model_tags = json.load(file)

    print(model_tags)

    img_size = model_tags['image_size'].strip('()').split(',')
    img_size = list(map(int, img_size))

    print(img_size)

    print("Model dir")
    print(os.environ['AZUREML_MODEL_DIR'])

    model_path = Model.get_model_path(model_tags['name'], model_tags['version']) 
    model = load_model(model_path)
    

def run(input_data):
    data = json.loads(input_data)['data']

    base64_decoded = base64.b64decode(data.strip())
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)/255

    print("input shape")
    print(image_np.shape)

    image_np = resize(image_np, img_size)

    print("input shape")
    print(image_np.shape)

    image_np = reshape(image_np, (-1, image_np.shape[0], image_np.shape[1], image_np.shape[2]))

    print("input shape")
    print(image_np.shape)

    prediction = model.predict(image_np)

    return json.dumps(prediction.tolist())