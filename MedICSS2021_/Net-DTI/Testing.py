"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Testing.py -h`

"""

from unicodedata import decimal
import numpy as np
import os
import time
import tensorflow as tf

from scipy.io import savemat, loadmat
from sympy import arg

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, load_nii_image, unmask_nii_data, loss_funcs, fetch_train_data_MultiSubject
from utils.model import parser


# Get parameter from command-line input
def test_model(args):
    test_subjects = args.test_subjects[0]
    nDWI = args.DWI
    scheme = args.scheme
    mtype = args.model
    lr = args.lr
    kernels = args.kernels
    layer = args.layer
    label_type = args.label_type
    patch_size = args.patch_size

    # Constants
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    decay = 0.1

    # Parameter name definition
    savename = str(nDWI)+ '-'  + scheme + '-' + args.model + '-' + str(layer) + 'layer'

    # Define the adam optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Load testing data
    mask = load_nii_image('datasets/mask/mask_' + test_subjects + '.nii')
    tdata = loadmat('datasets/data/' + test_subjects + '-' + str(nDWI) + '-' + scheme + '.mat')['data']

    test_shape = args.test_shape
    # tdata.shape = (x,y,z,ndwi)
    if test_shape is None:
        # test_shape = tdata.shape[1:4]
        # change the test_shape to include the (x,y,z,ndwi) rather than (y,z,ndwi)
        test_shape = tdata.shape[0:4]

    print(test_shape)

    # Define the model
    print('tdata shape: ' + str(tdata.shape))
    model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape)
    model.model(adam, loss_func, patch_size)
    model.load_weight(savename)

    # Get the weights
    weights = model._model.layers[1].get_weights()

    # Predict on the test data.
    if mtype[:6] == 'conv3d':
        shape = tdata.shape
        tdata = tf.expand_dims(tdata, 0)
        pred = model.predict(tdata)
        print(pred.shape)
        pred = tf.squeeze(pred, [0])
        print(pred.shape)
    else:
        pred = model.predict(tdata)
        print(pred.shape)

    # Evluate on the test data
    tlabel = loadmat('datasets/label/' + test_subjects + '-' + ''.join(ltype) +'-' + str(nDWI) + '-' + scheme + '.mat')['label']

    pred = repack_pred_label(pred, mask, mtype, len(ltype))
    print(calc_RMSE(pred, tlabel, mask, percentage=False, model=mtype))

    # Save estimated measures to /nii folder as nii image
    os.system("mkdir -p nii")

    for i in range(len(ltype)):
        data = pred[..., i]
        filename = 'nii/' + test_subjects + '-' + ltype[i] + '-' + savename + '.nii'

        data[mask == 0] = 0
        save_nii_image(filename, data, 'datasets/mask/mask_' + test_subjects + '.nii', None)

if __name__ == '__main__':
  args = parser().parse_args()
  test_model(args)