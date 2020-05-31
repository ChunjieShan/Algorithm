#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python3
# -*- coding: utf8 -*-

import cv2
import tensorflow as tf
import numpy as np



CATEGORIES = ["万用表", "示波器", "电阻", "焊台"]# will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), 3)  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.


# In[14]:


model = tf.keras.models.load_model("./mobilenet_14.h5")


# In[15]:


prediction = model.predict([prepare('../pictures/')])
print(CATEGORIES[int(np.argmax(prediction))])
