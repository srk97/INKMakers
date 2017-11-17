
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import requests

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#from read_data import read_data
import cv2
from map_data import labels_map
import threading


#train_data,train_labels = read_data()
batch_size = 128
nb_classes = 43
nb_epoch = 25

#print(train_labels.shape)
#print(train_labels[39207])
# input image dimensions

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(train_labels,nb_classes)
#Y_test = np_utils.to_categorical(, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3,
                        border_mode='same',
                        input_shape=(28,28,3)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
'''
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
'''
'''
model.fit(train_data, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
'''          

model.load_weights('model_new.hdf5')

import matplotlib.pyplot as plt


from fetch_image import url_to_image
import json


def predict():
    threading.Timer(5.0, predict).start()
    testimg = url_to_image(
        'http://192.168.43.127:8090/?action=snapshot')

    testimg = np.reshape(testimg, (1, 28, 28, 3))
    label=np.argmax(model.predict(testimg)[0][:])
    print(json.dumps(labels_map[label]))
    requests.get("http://192.168.43.127/say?w="+labels_map[label])


predict()

