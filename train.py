#!/usr/bin/python3
# -*- coding: utf8 -*-

from network import build_model
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
# import tensorflowjs as tfjs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers

# gpu = tf.compat.v1.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

FOLDER = '../data'
train_data_dir = "../data/train/"
val_data_dir = "../data/val/"
train_samples_num = 2250 # The amount of the photos in traning set.
val_samples_num = 250 # The amount of the photos in validation set.
IMG_W, IMG_H, IMG_CH = 150, 150, 3 # The size of the single photo.
save_folder = '../data/bottleneck' # Where to save bottleneck.
batch_size = 16 # Batch size.
epochs = 50 # Training epoch.
class_num = 5
model = build_model((IMG_W, IMG_H, IMG_CH), class_num)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(IMG_W, IMG_H),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
# bottleneck_features_train = model.predict(train_generator, 2700)
# np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                target_size=(IMG_W, IMG_H),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=False)
# bottleneck_features_validation = model.predict(generator, 300)
# np.save(open('bottleneck_features_validation.npy', 'w'),
#         bottleneck_features_validation)

history_ft = model.fit(
    train_generator, # 数据流
    steps_per_epoch=train_samples_num // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_samples_num // batch_size)

model.save("../models/Simple_classifier.h5")
# tfjs.converters.save_keras_model(model, "./models/model.json")

# train_data = np.load(os.path.join(save_folder,
#                                   'bottleneck_features_train.npy'))
# train_labels = np.array([0] * 450 + [1] * 450 + [2] * 450 + [3] * 450 +
#                         [4] * 450 + [5] * 450)
#
# train_labels = to_categorical(train_labels, class_num)
#
# validation_data = np.load(
#     os.path.join(save_folder, 'bottleneck_features_val.npy'))
# validation_labels = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50 +
#                              [4] * 50 + [5] * 50)
# validation_labels = to_categorical(validation_labels, class_num)
# # print(validation_labels)
#
# history_ft = model.fit(train_data,
#                        train_labels,
#                        epochs=epochs,
#                        batch_size=batch_size,
#                        validation_data=(validation_data, validation_labels))
