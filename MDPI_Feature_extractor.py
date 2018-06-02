'''
This code will be used to do feature extraction using CNN
'''


from __future__ import print_function
from keras.models import Sequential, Model
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import scipy.io
from keras.utils import to_categorical


# main code
if __name__ == '__main__':


    print('Code starts here: Feature generation from pretrained CNNs')

    # General parameters

    # read data
    mat1 = scipy.io.loadmat('data/Toronto.mat')
    mat2 = scipy.io.loadmat('data/Vaihangen.mat')

    Images1 = mat1['data']
    y1 = mat1['labels']


    Images2 = mat2['data']
    y2 = mat2['labels']

    num_classes = y1.max()
    y1 = y1-1
    y1 = to_categorical(y1,num_classes)
    y2 = y1

    # Define network
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

    # Generate Features and save to a file
    X1 = np.empty((0,4096))
    for idx1 in range(Images1.shape[3]):
        x1 = image.img_to_array(Images1[:,:,:,idx1])
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)
        fc1 =model.predict(x1)
        X1 = np.concatenate((X1,fc1),axis=0)
        print(idx1)


    X2 = np.empty((0,4096))
    for idx2 in range(Images2.shape[3]):
        x2 = image.img_to_array(Images2[:,:,:,idx2])
        x2 = np.expand_dims(x2, axis=0)
        x2 = preprocess_input(x2)
        fc1 =model.predict(x2)
        X2 = np.concatenate((X2,fc1),axis=0)
        print(idx2)


    np.savez('data/Trento_Potsdam.npz', X1=X1, y1=y1, X2=X2, y2=y2 )





