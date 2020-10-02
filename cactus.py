#Loading require library
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
#print(os.listdir("/Users/napasin_h/Desktop/aerial-cactus-identification"))
import numpy as np

train_dir = "/Users/napasin_h/Desktop/aerial-cactus-identification/train"
test_dir = "/Users/napasin_h/Desktop/aerial-cactus-identification/test"
train = pd.read_csv('/Users/napasin_h/Desktop/aerial-cactus-identification/train.csv')
df_test = pd.read_csv('/Users/napasin_h/Desktop/aerial-cactus-identification/sample_submission.csv')

#Data preparation
    #Read picture file
    #Decode JPEG content to RGB pixels
    #Convert this into floating tensors
    #Rescale pixels values(0-255) to [0, 1] interval
datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 150

#create a dataframe using pandas and text files provided,
#and create a meaningful dataframe with columns having file name
#(only the file names, not the path) and other classes to be used by the model
#Change only 1 column to string
train['has_cactus'] = train['has_cactus'].astype(str)
train_generator = datagen.flow_from_dataframe(dataframe = train[:15001], directory = train_dir, x_col = 'id',
                                             y_col = 'has_cactus', class_mode ='binary', batch_size = batch_size,
                                             target_size = (150, 150))

validation_generator=datagen.flow_from_dataframe(dataframe = train[15000:], directory = train_dir, x_col = 'id',
                                                y_col = 'has_cactus', class_mode = 'binary', batch_size = 50,
                                                 target_size=(150, 150))
