import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers


def build_model(input_shape, class_num):
    model = Sequential()
    model.add(
        Conv2D(256, (3, 3),
               strides=(2, 2),
               padding='same',
               input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(Conv2D(96, (1, 1), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(Conv2D(128, (1, 1), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    # model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-2, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    model = build_model((224, 224, 3), 6)
