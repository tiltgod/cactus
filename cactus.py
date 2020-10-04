#Loading require library
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
from IPython.display import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
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
#add arguement 'validation_split' for spliting data automatically and add 'subset' for define 'training' or 'validation'
datagen = ImageDataGenerator(rescale = 1./255)
#datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)
batch_size = 150

#create a dataframe using pandas and text files provided,
#and create a meaningful dataframe with columns having file name
#(only the file names, not the path) and other classes to be used by the model
#Change only 1 column to string
train['has_cactus'] = train['has_cactus'].astype(str)
#split train and validate dataset
train_generator = datagen.flow_from_dataframe(dataframe = train[:15001], directory = train_dir, x_col = 'id',
                                             y_col = 'has_cactus', class_mode ='binary', batch_size = batch_size, target_size = (150, 150))

validation_generator=datagen.flow_from_dataframe(dataframe = train[15000:], directory = train_dir, x_col = 'id',
                                                y_col = 'has_cactus', class_mode = 'binary', batch_size = 50, target_size=(150, 150))
#try validation split in ImageDataGenerator
#test_validation_split = datagen.flow_from_dataframe(dataframe = train, directory = train_dir, x_col = 'id',
                                             #y_col = 'has_cactus', class_mode ='binary', batch_size = batch_size ,subset = 'validation', target_size = (150, 150))

#Convolutions are defined on two key parameters
    #The size of patches that are extracted from input feature map..ie here 3x3
    #The number of filters computed from convolutions

#Building our model
#5 Conv2D + Maxpooling2D stages with relu activation function.
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(Conv2D(128, (3, 3),activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3),activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

#Compiling our model
    #loss: we will set our loss as binary_crossentropy since we are attacking a binary classification problem
    #optimizer : optimizers shape and mold your model into its most accurate possible form by futzing with the weights.
    #metrics : This is the evaluation criteria that we choose to evaluate our model
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(),metrics=['acc'])

#Fitting our model
epochs=10
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)

#Evaluating our model

#Improving our model using VGG16
model_vg=VGG16(weights='imagenet',include_top=False)
model_vg.summary()

#Extracting features using VGG16
def extract_features(directory,samples,df):


    features=np.zeros(shape=(samples,4,4,512))
    labels=np.zeros(shape=(samples))
    generator=datagen.flow_from_dataframe(dataframe=df,directory=directory,x_col='id',
                                            y_col='has_cactus',class_mode='other',batch_size=batch_size,
                                            target_size=(150,150))
    i=0
    for input_batch,label_batch in generator:
        feature_batch=model_vg.predict(input_batch)
        features[i*batch_size:(i+1)*batch_size]=feature_batch
        labels[i*batch_size:(i+1)*batch_size]=label_batch
        i+=1
        if(i*batch_size>samples):
            break
    return(features,labels)

train.has_cactus=train.has_cactus.astype(int)
features,labels=extract_features(train_dir,17500,train)
train_features=features[:15001]
train_labels=labels[:15001]

validation_features=features[15000:]
validation_labels=labels[15000:]

#Reshaping our features to feed into our dense layers
#df_test.has_cactus=df_test.has_cactus.astype(str)
test_features,test_labels=extract_features(test_dir,4000,df_test)
train_features=train_features.reshape((15001,4*4*512))
validation_features=validation_features.reshape((2500,4*4*512))
test_features=test_features.reshape((4000,4*4*512))

#Define a densely connected network
model=models.Sequential()
model.add(layers.Dense(212,activation='relu',kernel_regularizer=regularizers.l1_l2(.001),input_dim=(4*4*512)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.rmsprop(),loss='binary_crossentropy',metrics=['acc'])
history=model.fit(train_features,train_labels,epochs=30,batch_size=15,validation_data=(validation_features,validation_labels))

#making prediction
y_pre=model.predict_proba(test_features)

#making submission
df=pd.DataFrame({'id':df_test['id'] })
df['has_cactus']=y_pre
df.to_csv("submission.csv",index=False)
