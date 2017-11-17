
from readTrafficSigns import readTrafficSigns
import numpy as np
import os

def read_data():

    train_images,train_labels=readTrafficSigns(os.getcwd()+'/GTSRB/Final_Training/Images')

    train_labels=np.asarray(train_labels)
    train_images=np.asarray(train_images)

    return train_images,train_labels