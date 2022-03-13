"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Testing.py -h`

"""


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
                  MRIModel, load_nii_image, loss_funcs, fetch_test_data
from utils.model import parser


# Get parameter from command-line input
def test_model(args):
    test_subjects = args.test_subjects[0]
    nDWI = args.DWI
    mtype = args.model
    lr = args.lr
    kernels = args.kernels
    layer = args.layer
    label_type = args.label_type
    patch_size = args.patch_size
    base = args.base
    loss = args.loss
    test_shape = args.test_shape

    # Constants
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    lsavename = ''.join(ltype)
    # Parameter name definition
    if mtype == 'fc1d':
        patch_size = 1
    savename = str(nDWI) + '-' + args.model + '-' + \
        'patch' + '_' + str(patch_size) + \
        '-base_' + str(base) + \
        '-layer_' + str(layer)
    print(savename)

    # Load testing data
    mask = load_nii_image('datasets/mask/mask_' + test_subjects + '.nii')
    tdata = fetch_test_data(test_subjects, mask, nDWI, mtype, patch_size=patch_size)
    if test_shape is None:
        test_shape = tdata.shape[1:4]
        # change the test_shape to include the (x,y,z,ndwi) rather than (y,z,ndwi)
        # test_shape = tdata.shape[0:4]
    print(test_shape)

    # Define the model
    model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape)
    # Define the adam optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.model(adam, loss_funcs[loss], patch_size)
    model.load_weight(savename)

    # Get the weights
    weights = model._model.layers[1].get_weights()

    # Predict on the testset
    pred = model.predict(tdata)
    print('testing data shape: ' + str(tdata.shape))
    print('prediction has shape: ' + str(pred.shape))
    # Evluate on the test data
    tlabel = loadmat('datasets/label/' + test_subjects + '_' + lsavename + '.mat')['label']
    # repack the pred into suitable shape
    pred, masked_voxel = repack_pred_label(pred, mask, mtype, len(ltype))
    print('prediction after repack has shape: ' + str(pred.shape))

    print('the RMSE loss is: ' + str(calc_RMSE(pred, tlabel, mask, masked_voxel,percentage=False, model=mtype)))
    # if mtype == 'fc1d':
    #     print('the RMSE loss is: ' + str(calc_RMSE(pred, tlabel, mask, percentage=False, model=mtype)))
    # elif mtype == 'conv2d':
    #     shrinkedmask = mask[base:-base,base:-base,base:-base]
    #     x,y,z = shrinkedmask.shape
    #     temppred = pred[:x,:y,:z]
    #     temptlabel = tlabel[:x,:y,:z]
    #     print('the RMSE loss is: ' + str(calc_RMSE(temppred, temptlabel, shrinkedmask, percentage=False, model=mtype)))

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