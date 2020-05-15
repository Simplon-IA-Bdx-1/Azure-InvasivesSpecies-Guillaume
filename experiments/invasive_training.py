import pandas as pd
from pandas import DataFrame
from os import path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ["MPLBACKEND"] = "PS"
#import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.applications.vgg16 import VGG16
from azureml.core import Run
from azureml.core import Model
from sklearn.metrics import roc_curve, auc, accuracy_score

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--img_size', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--transfer_learning', default='false')

args = parser.parse_args()

run = Run.get_context()

#print(os.listdir())

#dataset = "./data"
print(run.input_datasets['images'])
train_img_dir = run.input_datasets['images']
labels_dataset = run.input_datasets['labels']
#fulltrain_dir = path.join(dataset,"train")
#test_dir = path.join(dataset,"test")

#labels = pd.read_csv(label_filename, index_col=0)
labels = labels_dataset.to_pandas_dataframe()
labels['filename'] = labels['name'].astype(str) + ".jpg"

SEED=42
TARGET_SIZE=(args.img_size, args.img_size)
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
tranfer_learning == args.transfer_learning

run.log('image_size', TARGET_SIZE)
run.log('batch_size', BATCH_SIZE)
run.log('epochs', EPOCHS)

print(labels.head())

labels_train,labels_valid = train_test_split(labels, test_size=0.2, random_state=SEED)

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.05,
                                   zoom_range=[0.7,0.9],
                                   rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen_noaugment = ImageDataGenerator(rescale=1./255)

options = {'directory': train_img_dir,
           'x_col': "filename",
           'y_col': "invasive",
           'batchsize': BATCH_SIZE,
           'class_mode': "raw",  
           'target_size': TARGET_SIZE,
           'seed': SEED
          }

print(os.listdir(train_img_dir))

train_generator = train_datagen.flow_from_dataframe(dataframe=labels_train, shuffle=True, **options)
valid_generator = train_datagen.flow_from_dataframe(dataframe=labels_valid, shuffle=False, **options)
# noaugment_generator = train_datagen_noaugment.flow_from_dataframe(dataframe=labels_train, shuffle=False, **options)

def build_model(size=(64,64)):
    return Sequential([
        Input(shape=(size[0],size[1],3)),
        Conv2D(16,(3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32,(3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64,(3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128,(3,3)),
        Activation('relu'),
        GlobalMaxPooling2D(),
        Dropout(rate=0.2),
        Dense(16),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ])

def build_VGG16(size=(64,64)):
    model = VGG16(weights= 'imagenet', include_top=False, input_shape=(size[0],size[1],3))
    
    for layer in model.layers:
        layer.trainable = False
        
    output = model.output
    output = Flatten()(output)
    output = Dropout(rate=0.5)(output)
    output = Dense(128)(output)
    output = Activation('relu')(output)
    output = Dropout(rate=0.5)(output)
    output = Dense(1)(output)
    output = Activation('sigmoid')(output)
    
    model = Model(model.input, output)
    return model

loss='binary_crossentropy'
metrics=[AUC()]
LEARNING_RATE = 0.01
#EPOCHS = 20

if args.transfer_learning == 'vgg16':
    model = build_VGG16(size=TARGET_SIZE)
else:
    tranfer_learning == 'false'
    model = build_model(size=TARGET_SIZE)
model.compile(loss=loss, optimizer='adamax', metrics=metrics)
#model.summary()

history = model.fit_generator(train_generator, epochs=EPOCHS, validation_data=valid_generator, verbose=0)

def plot_history(history, metrics=['loss'], val=False, shape=None, logy=False):
    fig = plt.figure(figsize=(15,8))
    if not isinstance(logy, list):
        logy = [logy] * len(metrics)
    df = DataFrame(history.history)
    if shape is None:
        shape = (1,len(metrics))
    
    for i, metric in enumerate(metrics):
        cols = [metric]
        if val:
            cols.append('val_' + metric)
        ax = fig.add_subplot(shape[0],shape[1],i+1)
        df[cols].plot(ax=ax, logy=logy[i])
        ax.grid(True)
        ax.set_title(f'Model performance throughout training ({metric})')
        ax.set_xlabel('epoch')
    plt.show()
    return fig

hist_plot = plot_history(history, val=True, metrics=['loss', 'auc'])
# plt.savefig('history.png')
run.log_image('History', plot=hist_plot)



y_valid = labels_valid['invasive']
y_pred_valid = model.predict_generator(valid_generator)

def plot_roc_curve(fpr, tpr, thresholds):
    fig = plt.figure(figsize=(14,6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.plot(fpr, thresholds, 'r--', label='threshold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])

    plt.legend(loc="lower right")
    return fig

fpr, tpr, thresholds = roc_curve(y_valid, y_pred_valid)
roc_plot = plot_roc_curve(fpr, tpr, thresholds)
run.log_image('ROC', plot=roc_plot)
auc_score = auc(fpr, tpr)

run.log("AUC", auc_score)

model.save('my_model.hdf5')
run.upload_file('model', path_or_stream='my_model.hdf5')
run.register_model(model_path="model",
               model_name="InvasiveCNN",
               tags={'AUC': auc_score, 'transfer-learning': args.transfer_learning, 'image_size': str(TARGET_SIZE), 'batch_size': BATCH_SIZE, 'epochs': EPOCHS},
               description="simple CNN",
               model_framework=Model.Framework.TENSORFLOW,
               model_framework_version=tf.__version__,
               )
