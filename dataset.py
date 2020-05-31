#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = '/home/rick/Computer_Competition/data/train/'

CATEGORIES = [
    "Resistance", "Capacitance", "Inductance", "Solder", "Multimeter",
    "Oscilloscope"
]

for category in CATEGORIES: # do dogs and cats
    path = os.path.join(DATADIR, category) # create path to dogs and cats
    for img in os.listdir(path): # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path, img)) # convert to array

training_data = []

IMG_SIZE = 224


def create_training_data():
    for category in CATEGORIES: # do dogs and cats

        path = os.path.join(DATADIR, category) # create path to dogs and cats
        class_num = CATEGORIES.index(
            category) # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), 3)
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            # except OSError as e:
            #     print("OSErrroBad img most likely", e, os.path.join(path, img))
            # except Exception as e:
            #     print("general exception", e, os.path.join(path, img))


create_training_data()

print("The length of training data:", len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
