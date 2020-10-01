import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as pimg
import seaborn as sns
import math
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalMaxPooling2D
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler

import model
np.random.seed(12)
tf.random.set_seed(12)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(GlobalMaxPooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)
#model.summary()

class CactusConfig(Config):
    """docstring for CactusConfig."""
    name = Cactus
    initial_rate = 0.001
    drop = 0.5
    epoch_drop = 10.0
    epochs = 30
    batch_size = 32
    img_width  = 32
    img_height = 32
    input_shape = (img_width, img_height, 3)
    optimizer = optimizers.Adam(lr=1e-3)

class CactusDataset(Dataset):
    """docstring for CactusDataset"""
    main_df = args.main_df
    sub_df = args.sub_df
    target_size = (config.img_width, config.img_height)

    def split_data()
        train_df, val_df = train_test_split(main_df, test_size=0.25, stratify= main_df['has_cactus'], shuffle=True, random_state=12)
        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        total_train = train_df.shape[0]
        total_val = val_df.shape[0]
        return total_train, total_val

    def define_data_augment()
        # Define Data Augmentation
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        return train_datagen, val_datagen, test_datagen

    def convert_label()
        # Convert the data type of 'has_cactus' to str to allow the model to be trained.
        train_df['has_cactus'] = train_df['has_cactus'].astype(str)
        val_df['has_cactus'] = val_df['has_cactus'].astype(str)

def step_decay(epoch, config)
    lrate = config.INITIAL_RATE * math.pow(config.DROP, math.floor((config.EPOCHS) / config.EPOCH_DROP))
    return lrate

def scheduler()
    lrate = LearningRateScheduler(step_decay)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
    callback = [lrate, es]
    return callback

def train_model(config, dataset, model)
    callbacks = scheduler()

    history = model.fit(
    train_gen,
    epochs=config.EPOCHS,
    steps_per_epoch=dataset.total_train//config.BATCH_SIZE,
    validation_data=dataset.val_gen,
    validation_steps=dataset.total_val//config.BATCH_SIZE,
    callbacks=callbacks,)
    return history

import argparse
parser = argparse.ArgumentParser(description='Cactus identification')
parser.add_argument("command", metavar="<command>", help="'train', 'predict'")
parser.add_argument("--main_df", metavar="")
parser.add_argument("--sub_df", metavar="")
parser.add_argument("--training_set", metavar="")
parser.add_argument("--test_set", metavar="")
args = parser.parse_args()

if args.command == "train":
    assert args.training_set, "training_set require for training"
    assert args.test_set, "training_set require for testing"
    assert args.main_df, "main_df require for training"
    assert args.sub_df, "sub_df require for submission"
    config = CactusConfig()
    dataset = CactusDataset()

if args.command == "predict":
    assert args.training_set, "training_set require for training"
    assert args.test_set, "training_set require for testing"
    assert args.main_df, "main_df require for training"
    assert args.sub_df, "sub_df require for submission"
