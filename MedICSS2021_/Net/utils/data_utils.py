"""
Functions for Generating or save dataset.
"""

import os
import argparse
from re import L
from traceback import print_tb
import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from utils.nii_utils import load_nii_image, save_nii_image, mask_nii_data, unmask_nii_data

"""
Rewrite the data_utils, because we think the dataset generated having some issues
"""
def gen_base_datasets(path, subject, label_type, fdata=True, flabel=True):
    """
    Generate base Datasets. 
    Needs to run for all subjects first
    """
    # labesl for NODDI
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/mask-e.nii datasets/mask/mask_' + subject + '.nii')
    print('Generating for ' + subject + ' ...')

    # define the savename
    savename = ''.join(ltype)
        
    if fdata:
        data = load_nii_image(path + '/' + subject + '/diffusion.nii')
        print('base data dataset has shape: ' + str(data.shape))
        savemat('datasets/data/' + subject + '.mat', {'data':data})

    if flabel:
        mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
        label = np.zeros(mask.shape + (len(ltype),))
        for i in range(len(ltype)):
            filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
            label[:, :, :, i] = load_nii_image(filename)
        print('base label dataset has shape: ' + str(label.shape))
        savemat('datasets/label/' + subject + '_' + savename + '.mat', {'label':label})

def gen_fc1d_datasets(path, subject, patch_size, label_size, label_type, base=1, test=False):
    """
    Generate fc1d training Datasets.
    """
    print("Generating for " + subject + " ...")

    # ltype are a list of label types
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    savename = ''.join(ltype)
    patch_size = 1
        
    # load masked diffusion data
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    data = loadmat('datasets/data/' + subject + '.mat')['data']
    data = mask_nii_data(data, mask)
    print('training dataset has shape:' + str(data.shape))

    # load labels, without scaling
    labels = loadmat('datasets/label/' + subject + '_' + savename + '.mat')['label']
    labels = mask_nii_data(labels, mask)
    print('training label has shape:' + str(labels.shape))

    # save datasets
    # save train data
    savemat('datasets/data/' + subject + '-base' + str(base) + '-patches-1d-' + str(patch_size)\
            + '-' + str(label_size) + '-all.mat', {'data':data})
    # save train label
    savemat('datasets/label/' + subject + '-base' + str(base) + '-labels-1d-' + str(patch_size)\
                + '-' + str(label_size) + '-' + savename + '.mat', {'label':labels})



def gen_2d_patches(data, mask, size, stride):
    """
    gen patches
    """
    patches = []
    for layer in range(mask.shape[2]): # visiting the slice
        for x in np.arange(0, mask.shape[0], stride): # visiting the row
            for y in np.arange(0, mask.shape[1], stride): # visiting the column
                xend, yend = np.array([x, y]) + size # set the patch indices
                lxend, lyend = np.array([x, y]) + stride # set the indices for masking
                if mask[x:lxend, y:lyend, layer].sum() > 0: # justify if the voxel is masked
                    patches.append(data[x:xend, y:yend, layer, :]) # if masked, add the voxel patch
    print(np.array(patches).shape)
    return np.array(patches)

def gen_3d_patches(data, mask, size, stride):
    """
    gen patches
    """
    print(data.shape, mask.shape)
    patches = []
    for layer in np.arange(0, mask.shape[2], stride): # visiting the slice
        for x in np.arange(0, mask.shape[0], stride): # visiting the row
            for y in np.arange(0, mask.shape[1], stride): # visiting the column
                xend, yend, layerend = np.array([x, y, layer]) + size # set the patch indices
                lxend, lyend, llayerend = np.array([x, y, layer]) + stride # set the indices for masking
                if mask[x:lxend, y:lyend, layer:llayerend].sum() > 0: # justify if the voxel is masked
                    patches.append(data[x:xend, y:yend, layer: layerend, :]) # if masked, add the voxel patch
    print(np.array(patches).shape)
    return np.array(patches)

def gen_conv2d_datasets(path, subject, patch_size, label_size, label_type, base=1, test=False):
    """
    Generate Conv2D Datasets.
    """
    # ltype are a list of label types
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    savename = ''.join(ltype)
    # offset = base - (patch_size - label_size) / 2

    print("Generating for " + subject + " ...")
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    mask = mask[base:-base, base:-base, base:-base]
    print('mask has shape: ' + str(mask.shape))
    data = loadmat('datasets/data/' + subject + '.mat')['data']
    data = data[:, :, base:-base, :]
    print('data has shape: ' + str(data.shape))
    labels = loadmat('datasets/label/' + subject + '_' + savename + '.mat')['label']
    labels = labels[base:-base, base:-base, base:-base, :]
    print('label has shape: ' + str(labels.shape))
    
    # if offset:
    #     data = data[offset:-offset, offset:-offset, :, :12]

    patches = gen_2d_patches(data, mask, patch_size, label_size)
    print('saved patches has shape: ' + str(patches.shape))
    savemat('datasets/data/' + subject + '-base' + str(base) + '-patches-2d-' + str(patch_size)\
        + '-' + str(label_size) + '-all.mat', {'data':patches})

    labels = gen_2d_patches(labels, mask, label_size, label_size)
    print('svaed labels has shape: ' + str(labels.shape))
    savemat('datasets/label/' + subject + '-base' + str(base) + '-labels-2d-' + str(patch_size)\
            + '-' + str(label_size) + '-' + savename + '.mat', {'label':labels})

def gen_conv3d_datasets(path, subjects, patch_size, label_size, base=1, test=False):
    """
    Generate Conv3D Datasets.
    """
    # offset = base - (patch_size - label_size) / 2
    for subject in subjects:
        
        print("Generating for " + subject + " ...")

        labels = loadmat('datasets/label/' + subject+ '.mat')['label']
        labels = labels[base:-base, base:-base, base:-base, :]
        mask = load_nii_image(path + '/' + subject + '/mask-e.nii')
        mask = mask[base:-base, base:-base, base:-base]
        
        data = loadmat('datasets/data/' + subject + '.mat')['data']
        # data = data[base:-base, base:-base, base:-base, :]

        # if offset:
        #     data = data[offset:-offset, offset:-offset, offset:-offset, :12]

        patches = gen_3d_patches(data, mask, patch_size, label_size)
        patches = patches.reshape(patches.shape[0], -1)
        savemat('datasets/data/' + subject + '-base' + str(base) + '-patches-3d-' + str(patch_size)\
            + '-' + str(label_size) + '-all.mat', {'data':patches},  format='4')

        labels = gen_3d_patches(labels, mask, label_size, label_size)
        savemat('datasets/label/' + subject + '-base' + str(base) + '-labels-3d-' + str(patch_size)\
                + '-' + str(label_size) + '-all.mat', {'label':labels})

        print(patches.shape)
        print(labels.shape)

def fetch_train_data(subjects, ndwi, model, label_type, patch_size=3, label_size=1, base=1,
                whiten=True, combine=None):
    """
    #Fetch train data.
    """
    data_s = None
    labels = None
    
    if model[:6] == 'conv2d':
        filename = '-2d-' + str(patch_size) + '-' + str(label_size)
    elif model[:6] == 'conv3d':
        filename = '-3d-' + str(patch_size) + '-' + str(label_size)
    elif model[:4] == 'fc1d':
        patch_size = 1
        filename = '-1d-' + str(patch_size) + '-' + str(label_size)
    
    print(label_type)
    if label_type == ['N']:
        ltype = ['NDI']
    elif label_type == ['O']:
        ltype = ['ODI']
    elif label_type == ['F']:
        ltype = ['FWF']
    elif label_type == ['A']:
        ltype = ['NDI' , 'ODI', 'FWF']
    
    savename = ''.join(ltype)

    for subject in subjects:
        label = loadmat('datasets/label/' + subject + '-base' + str(base) + '-labels' + filename + '-' + savename + '.mat')['label']
        data = loadmat('datasets/data/' + subject + '-base' + str(base) + '-patches' + filename + '-' + 'all.mat')['data']

        for i in range(label.shape[0]):
            if np.isnan(label[i]).any():
                label[i] = 0
                data[i] = 0

        if data_s is None:
            data_s = data
            labels = label
        else:
            data_s = np.concatenate((data_s, data), axis=0)
            labels = np.concatenate((labels, label), axis=0)
        

    data = np.array(data_s)
    label = np.array(labels)

    # needs to unsqueeze it to tensor
    if model[:6] == 'conv3d':
        data = data.reshape(data.shape[0], 3, 3, 3, -1)
    
    # Select the inputs.
    if combine is not None:
        data = data[..., combine == 1]
    else:
        data = data[..., :ndwi]

    # whiten the data.
    if whiten:
        data = data / data.mean() - 1.0

    return data, label

def fetch_test_data(subject, mask, ndwi, model, patch_size=3, label_size=1, base=1,
                    whiten=True, combine=None):
    """
    Fetch test data.
    """   
    data = loadmat('datasets/data/' + subject + '.mat')['data']

    # Select the inputs.
    if combine is not None:
        data = data[..., combine == 1]
    else:
        data = data[..., :ndwi]

    # Whiten the data.
    if whiten:
        data = data / data[mask > 0].mean() - 1.0

    # Reshape the data to suit the model.
    if model[:6] == 'conv3d':
        data = np.expand_dims(data, axis=0)
    elif model[:6] == 'conv2d':
        data = data.transpose((2, 0, 1, 3))

    return data

def shuffle_data(data, label):
    """
    Shuffle data.
    """
    size = data.shape[-1]
    datatmp = np.concatenate((data, label), axis=-1)
    np.random.shuffle(datatmp)
    return datatmp[..., :size], datatmp[..., size:]

def repack_pred_label(pred, mask, model, ntype, segment=False):
    """
    Get.
    """
    masked_voxel = 0
    if model[7:13] == 'single':
        label = np.zeros(mask.shape + (1,))
    else:
        label = np.zeros(mask.shape + (ntype,))
    
    if model[:6] == 'conv2d':
        unpadded_label = pred.transpose(1, 2, 0, 3)
        unpadded_label = unpadded_label[:,:,:48,:]
        shrinked_mask = mask[1:-1,1:-1,1:-1]
        masked_unpadded_label = unpadded_label[shrinked_mask>0]
        print('masked unpadded label has shape: ' + str(masked_unpadded_label.shape))
        masked_voxel = masked_unpadded_label.shape[0]
        label[1:-1, 1:-1, :, :] = pred.transpose(1, 2, 0, 3)
        # just for test
        # label[1:-1, 1:-1, 1:-1, :] = unpadded_label
        masked_label = label[mask>0]
        print('masked padded label has shape: ' + str(masked_label.shape))
    elif model[:6] == 'conv3d':
        label[1:-1, 1:-1, 1:-1, :] = pred
    else:
        label = pred.reshape(label.shape)
        masked_label = label[mask>0]
        print('masked label has shape: ' + str(masked_label.shape))
        masked_voxel = masked_label.shape[0]

    return label, masked_voxel