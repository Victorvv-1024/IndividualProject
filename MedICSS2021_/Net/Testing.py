"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python3 Testing.py -h`

Usage:
1. To test a fc1d model:
    
    python3 Testing.py --path $DataDir --subjects s01_still --label_type label --fc1d 
  
2. To test a 2D CNN model:
    
    python3 Testing.py --path $DataDir --subjects s01_still --label_type label --conv2d 

3. To test a 3D CNN model:
    
    python3 Testing.py --path $DataDir --subjects s01_still --label_type label --conv3d 
"""


import numpy as np
import os
import time
import tensorflow as tf

from scipy.io import savemat, loadmat
from sympy import arg

from tensorflow.keras.optimizers import SGD, Adam

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, load_nii_image, loss_funcs, fetch_test_data, \
                  calc_ssim
from utils.model import parser


# Get parameter from command-line input
def test_model(args):
    # get the test paramters
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

    # determin the input DWI volumes using a scheme file
    combine = None
    movefile = args.movefile
    if movefile is not None:
        # file = open(movefile,'r')
        # combine = np.array([int(num) for num in file.readline().split(' ')[:-1]])
        combine = np.array([int(float(num)) for num in movefile])
        nDWI = combine.sum() # update the input size

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
    
    if label_type != ['A']:
        out = 1 # specify the output dimension of the network
    else: out = 3

    # Parameter name definition
    if mtype == 'fc1d':
        patch_size = 1

    savename = str(nDWI) + '-' + args.model + '-' + \
           'patch' + '_' + str(patch_size) + \
           '-base_' + str(base) + \
           '-layer_' + str(layer)+ \
           '-label_' + lsavename + 'synthetic'
    print(savename)

    mask = load_nii_image('datasets/mask/mask_' + test_subjects + '.nii')
    tdata = fetch_test_data(test_subjects, mask, nDWI, mtype, patch_size=patch_size, combine=combine)
    if test_shape is None:
        test_shape = tdata.shape[1:4]

    # Define the model
    model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape, out=out)
    # Define the adam optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.model(adam, loss_funcs[loss], patch_size)
    model.load_weight(savename)

    # Get the weights
    weights = model._model.layers[1].get_weights()

    # Predict on the testset
    # pred[0] is NDI pred[1] is ODI and pred[2] is FWF
    pred = model.predict(tdata)

    # get the ground truth label (NOT SYNTHETIC)
    if len(ltype) == 3:
        ndi = loadmat('datasets/label/' + test_subjects + '_NDI.mat')['label'][:,:,:,0]
        odi = loadmat('datasets/label/' + test_subjects + '_ODI.mat')['label'][:,:,:,0]
        fwf = loadmat('datasets/label/' + test_subjects + '_FWF.mat')['label'][:,:,:,0]
    else:
        tlabel = loadmat('datasets/label/' + test_subjects + '_' + lsavename + '.mat')['label']
    print('ndi label has shape ' + str(ndi.shape))

    # repack the pred into suitable shape
    pred = repack_pred_label(pred, mask, mtype, len(ltype))
    print('prediction after repack has shape: ' + str(pred.shape))

    # Evaluate the prediction by RMSE and SSIM
    RMSE = []
    SSIM = []
    if len(ltype) == 3:
        pred_ndi = pred[:,:,:,0]
        print('pred ndi has shape: ' + str(pred_ndi.shape))
        ndi_loss = calc_RMSE(pred[:,:,:,0], ndi, mask, percentage=False, model=mtype)
        print(ndi_loss)
        RMSE.append(ndi_loss)
        odi_loss = calc_RMSE(pred[:,:,:,1], odi, mask, percentage=False, model=mtype)
        RMSE.append(odi_loss)
        fwf_loss = calc_RMSE(pred[:,:,:,2], fwf, mask, percentage=False, model=mtype)
        RMSE.append(fwf_loss)

        SSIM.append(round(calc_ssim(pred[:,:,:,0], ndi),3))
        SSIM.append(round(calc_ssim(pred[:,:,:,1], odi),3))
        SSIM.append(round(calc_ssim(pred[:,:,:,2], fwf),3))
    else:
        RMSE.append(round(calc_RMSE(pred, tlabel, mask, percentage=False, model=mtype),3))
        SSIM.append(round(calc_ssim(pred, tlabel),3))
    print('the RMSE loss is: ' + str(RMSE))
    print('the SSIM loss is: ' + str(SSIM))

    # Save estimated measures to /nii folder as nii image
    os.system("mkdir -p nii")

    for i in range(len(ltype)):
        data = pred[..., i]
        # update the savename to synthetic
        savename = str(nDWI) + '-' + args.model + '-' + \
           'patch' + '_' + str(patch_size) + \
           '-base_' + str(base) + \
           '-layer_' + str(layer)+ \
           '-label_' + ltype[i] + 'synthetic'
                    
        filename = 'nii/' + test_subjects + '-' + savename + '.nii'

        data[mask == 0] = 0
        save_nii_image(filename, data, 'datasets/mask/mask_' + test_subjects + '.nii', None)
    
    return RMSE, SSIM

if __name__ == '__main__':
  args = parser().parse_args()
  test_model(args)