"""
Module definition for network training and testing.
"""

import os
import argparse
from tabnanny import verbose
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv3D, ReLU, Flatten, Reshape, Conv2DTranspose, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras import backend as K


class MRIModel(object):
    """
    MRI models
    """

    _ndwi = 0 # the input size
    _single = False
    _model = None # the choice of the model
    _type = '' # the type of model, input from user
    _loss = [] # choice of loss function
    _kernel1 = 150 # kernel size
    _kernel2 = 200 # kernel size
    _kernel3 = 250 # kernel size
    _out = 3

    def __init__(self, ndwi=96, model='fc1d', layer=3, train=True, kernels=None, test_shape=None, out=3):
        """
        initialisation of MRI model class

        Args:
            ndwi (int, optional): the number of input size. Defaults to 96.
            model (str, optional): the choice of the model. Defaults to 'fc1d'.
            layer (int, optional): the number of hidden layers. Defaults to 3.
            train (bool, optional): if True, then do training. Else, do testing. Defaults to True.
            kernels (_type_, optional): a list of integers, that defines the kernel sizes for kernel1,2,3. Defaults to None.
            test_shape (_type_, optional): the shape of the test data. It should be a 3D. Defaults to None.
            out(int, optional): specify the output dimension of the model
        """        
        self._ndwi = ndwi
        self._type = model
        self._hist = None
        self._train = train
        self._layer = layer
        self._test_shape = test_shape
        self._model = None
        if kernels is not None:
            self._kernel1, self._kernel2, self._kernel3 = kernels
        self._out = out

    def _fc1d_model(self, patch_size):
        """
        Fully-connected 1d ANN model.
        """
        # the Input layer takes input of size (None, _ndwi)
        inputs = Input(shape=(self._ndwi,))
        # Define hidden layer
        hidden = Dense(self._kernel1, activation='relu')(inputs)
        for i in np.arange(self._layer  - 1):
            hidden = Dense(self._kernel1, activation='relu')(hidden)
        hidden = Dropout(0.1)(hidden)
        # Define output layer
        # The output size can be changed from 1 to 3
        outputs = Dense(self._out, name='output')(hidden)
        # change the activation at the output layer sig, gives a value between 0 and 1
        activation_layer = ReLU(max_value=1.0)
        outputs = activation_layer(outputs)

        self._model = Model(inputs=inputs, outputs=outputs)
        
    def _conv2d_model(self, patch_size):
        """
        Conv2d model.
        """
        # The input dimension for training is by default 3x3x96; where 3x3 suggests the patch size and 96 is the input size
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, self._ndwi))
        # The input dimension for testing is by default testshape[0]xtestshape[1]x96. Hence we can feed the image directly to the trained network
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, self._ndwi))
        # The first Convolutional layer, having kernel size 3x3
        hidden = Conv2D(self._kernel1, 3, strides=1, activation='relu', padding='valid')(inputs)
        # Define the hidden layer
        for i in np.arange(self._layer - 1):
            hidden = Conv2D(self._kernel1, 1, strides=1, activation='relu', padding='valid')(hidden)
        hidden = Dropout(0.1)(hidden)
        # Define output layer
        # The output size can be changed from 1 to 3
        outputs = Conv2D(self._out, 1, strides=1, padding='valid')(hidden)
        activation_layer = ReLU(max_value=1.0)
        outputs = activation_layer(outputs)

        self._model = Model(inputs=inputs, outputs=outputs)

    def _conv3d_model(self, patch_size):
        """
        Conv3D model.
        """
        # The input dimension for training is by default 3x3x3x96; where 3x3x3 suggests the patch size and 96 is the input size
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, patch_size, self._ndwi))
        # The input dimension for testing is testshape[0]xtestshape[1]xtestshape[2]x96. Hence we can feed the image directly to the trained network
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._ndwi))
        # The first Convolutional layer, having kernel size 3x3x3
        hidden = Conv3D(self._kernel1, 3, activation='relu', padding='valid')(inputs)
        # Define the hidden layer
        for i in np.arange(self._layer - 1):
            hidden = Conv3D(self._kernel1, 1, activation='relu', padding='valid')(hidden)
        hidden = Dropout(0.1)(hidden)
        # Define output layer
        # The output size can be changed from 1 to 3
        outputs = Conv3D(self._out, 1, padding='valid')(hidden)
        activation_layer = ReLU(max_value=1.0)
        outputs = activation_layer(outputs)
        
        self._model = Model(inputs=inputs, outputs=outputs)

    # The choice of models
    __model = {
        'fc1d' : _fc1d_model,
        'conv2d': _conv2d_model,
        'conv3d' : _conv3d_model,
    }

    def model(self, optimizer, loss, patch_size):
        """
        Generate the model
        Args:
            optimizer: the choice of optimizer for the model
            loss: the loss function of the model
            patch_size (int): the patch size, by default it is 3
        """
        # generate the model
        self.__model[self._type](self, patch_size)
        # gives the summary of the model, including number of trained parameters; output shapes at each layer and so on.
        self._model.summary()
        # compile the model. The metrics is accuracy because we are doing regression.
        self._model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _sequence_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):
        """
        sequentially train the model

        Args:
            data (ndarray): the fetched train data
            label (ndarray): the fetched train label
            nbatch (int): the number of batches per epoch. By default, it is 256
            epochs (int): the number of epoches. By default, it is 100
            callbacks: enables early stop
            shuffle (boolean): if True, shuffle the data. Else, not.
            validation_data (ndarray): the validation data
        """        
        validation_split = 0.0
        # if the validation data is none, we split the train data into train data and validation data
        if validation_data is None:
            validation_split = 0.2 # set the split to 0.5 becasue the number of training input is large

        # fit the model to the train dataset and validated against validation dataset
        self._hist = self._model.fit(data, label,
                                     batch_size=nbatch,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks,
                                     verbose=0)# set only when I want a silent environment
        self._loss.append(len(self._hist.history['loss']))
        self._loss.append(self._hist.history['loss'][-1])
        self._loss.append(None)
        self._loss.append(self._hist.history['accuracy'][-1])
        self._loss.append(None)

    # for each model type, use the corresponding train method
    __train = {
        'fc1d' : _sequence_train,
        'conv2d': _sequence_train,
        'conv3d' : _sequence_train,
    }

    def train(self, data, label, nbatch, epochs, callbacks, weightname,
              shuffle=True, validation_data=None):
        """
        Training on training datasets.
        """
        print("Training start ...")
        self.__train[self._type](self, data, label, nbatch, epochs,
                                 callbacks, shuffle, validation_data)
        # save the weight, hence the saved weight can be directly used when testing
        try:
            self._model.save_weights('weights/' + weightname + '.weights')
        except IOError:
            os.system('mkdir weights')
            self._model.save_weights('weights/' + weightname + '.weights')

        return self._loss

    def load_weight(self, weightname):
        """
        Load pre-trained weights.
        """
        self._model.load_weights('weights/' + weightname + '.weights').expect_partial()

    def predict(self, data):
        """
        Predict on test datas.
        """
        pred = self._model.predict(data)
        return pred

def parser():
    """
    Create a parser.
    """
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--train", help="Train the network", action="store_true")
    parser.add_argument("--model", help="Train model",
                        choices=['conv3d',  'fc1d', 'conv2d'], default='conv3d')
    parser.add_argument("--layer", metavar='l', help="Number of layers", type=int, default=3)
    parser.add_argument("--lr", metavar='lr', help="Learning rates", type=float, default=0.0001)
    parser.add_argument("--epoch", metavar='ep', help="Number of epoches", type=int, default=100)
    parser.add_argument("--kernels", help="The number of kernels for each layer", nargs='*',
                        type=int, default=None)
    parser.add_argument("--patch_size", metavar='ksize', help="Size of the kernels", type=int, default=3) #default patch_size is already 3

    
    # Specify train & test sets
    parser.add_argument("--train_subjects", help="Training subjects IDs", nargs='*')
    parser.add_argument("--test_subjects", help="Testing subject ID", nargs='*')
    parser.add_argument("--movefile", default=None)
    parser.add_argument("--DWI", metavar='N', help="Number of input DWI volumes", type=int, default=96)
    parser.add_argument("--batch", metavar='bn', help="Batch size", type=int, default=256)
    parser.add_argument("--base", metavar='base', help="choice of training data", type=int, default=1)
    parser.add_argument("--label_type", help="select which label to train. N for NDI, O for ODI and F for FWF; A for all.", 
                        choices=['N', 'O', 'F', 'A'], nargs=1)

    parser.add_argument("--loss", help="Set different loss functions", type=int, default=0)
    parser.add_argument("--test_shape", nargs='*', type=int, default=None)

    return parser
