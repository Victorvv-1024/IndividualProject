"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Training.py -h`

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
                                    

# from utils import MRIModel, parser, loss_funcs, fetch_train_data_MultiSubject
from utils import MRIModel, loss_funcs, fetch_train_data
from utils.model import parser


# Get parameter from command-line input
def train_network(args):
    train_subjects = args.train_subjects
    nDWI = args.DWI
    scheme = args.scheme
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

    # Parameter name definition
    # savename = str(nDWI)+ '-'  + scheme + '-' + args.model + '-' + str(layer) + 'layer'
    if mtype == 'fc1d':
        patch_size = 1
    savename = str(nDWI) + '-' + args.model + '-' + \
           'patch' + '_' + str(patch_size) + \
           '-base_' + str(base) + \
           '-layer_' + str(layer)

    # Constants
    types = ['NDI' , 'ODI', 'FWF']
    ntypes = len(types)
    decay =  1e-6

    shuffle = False
    # y_accuracy = None
    # output_accuracy = None
    # y_loss = None
    # output_loss = None
    # nepoch = None

    # Define the adam optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Train on the training data.
    if train:
        # Define the model.
        model = MRIModel(nDWI, model=mtype, layer=layer, train=train, kernels=kernels)

        model.model(adam, loss_funcs[loss], patch_size) # use the RMSE loss
        # model.model(adam,loss=MeanAbsoluteError(), patch_size=patch_size)

        data, label = fetch_train_data(train_subjects, nDWI, mtype,
                                       label_type=label_type,
                                       patch_size=patch_size,
                                       label_size=label_size,
                                       base=base,
                                       )
        reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, epsilon=0.0001)
        tensorboard = TensorBoard(histogram_freq=0)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.0000005)
#       [nepoch, output_loss, y_loss, output_accuracy, y_accuracy]
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