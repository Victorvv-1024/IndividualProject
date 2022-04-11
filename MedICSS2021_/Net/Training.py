"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python3 Training.py -h`

Usage:
1. To train a fc1d model:
    
    python3 Training.py --path $DataDir --subjects s01_still --label_type label --fc1d 
  
2. To train a 2D CNN model:
    
    python3 Training.py --path $DataDir --subjects s01_still --label_type label --conv2d 

3. To train a 3D CNN model:
    
    python3 Training.py --path $DataDir --subjects s01_still --label_type label --conv3d 
"""

import numpy as np
import os
import time
from matplotlib import pyplot as plt
from pip import main

from scipy.io import savemat

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping
from tensorflow import keras
                                    
from utils import MRIModel, loss_funcs, fetch_train_data
from utils.model import parser


# Get parameter from command-line input
def train_network(args):
    # the training parameters
    train_subjects = args.train_subjects
    nDWI = args.DWI
    mtype = args.model
    train = args.train
    lr = args.lr
    epochs = args.epoch
    kernels = args.kernels
    layer = args.layer
    label_type = args.label_type
    loss = args.loss
    batch_size = args.batch
    patch_size = args.patch_size
    label_size = patch_size - 2
    base = args.base

    # determin the input DWI volumes using a scheme file
    combine = None
    movefile = args.movefile
    if movefile is not None:
        file = open(movefile,'r')
        combine = np.array([int(num) for num in file.readline().split(' ')[:-1]]) # the scheme file
        nDWI = combine.sum() # update the input size
    print(nDWI)

    # Parameter name definition
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    lsavename = ''.join(ltype)
    if mtype == 'fc1d':
        patch_size = 1
    # savename = str(nDWI) + '-' + args.model + '-' + \
    #        'patch' + '_' + str(patch_size) + \
    #        '-base_' + str(base) + \
    #        '-layer_' + str(layer)+ \
    #        '-label_' + lsavename
    # update the savename to synthetic
    savename = str(nDWI) + '-' + args.model + '-' + \
           'patch' + '_' + str(patch_size) + \
           '-base_' + str(base) + \
           '-layer_' + str(layer)+ \
           '-label_' + lsavename + 'synthetic'
        
    if label_type != ['A']:
        out = 1 #specify the output dimension of the network
    else: out = 3

    # Constants
    types = ['NDI' , 'ODI', 'FWF']
    ntypes = len(types)
    decay =  1e-6

    shuffle = False

    # Define the adam optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Train on the training data.
    if train:
        # Define the model.
        model = MRIModel(nDWI, model=mtype, layer=layer, train=train, kernels=kernels, out=out)

        model.model(adam, loss_funcs[loss], patch_size) # use the RMSE loss, if loss=0

        data, label = fetch_train_data(train_subjects, nDWI, mtype,
                                       label_type=label_type,
                                       patch_size=patch_size,
                                       label_size=label_size,
                                       base=base,
                                       combine=combine)

        extractor = keras.Model(inputs=model._model.inputs,
                outputs=[layer.output for layer in model._model.layers])
        features = extractor(data)
        for f in features:
            print(f.shape)
        
        # define the early stop
        reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, epsilon=0.00001)
        tensorboard = TensorBoard(histogram_freq=0)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.0000005)

        result = model.train(data, label, batch_size, epochs,
                            [reduce_lr, tensorboard, early_stop],
                            savename, shuffle=not shuffle,
                            validation_data=None)
        history = model._hist
        return history
        
if __name__ == '__main__':
    args = parser().parse_args()
    history = train_network(args)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()