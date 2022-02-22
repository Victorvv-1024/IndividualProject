"""
Functions for Generating or save dataset.
"""

import os
from traceback import print_tb
import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from utils.nii_utils import load_nii_image, save_nii_image, mask_nii_data, unmask_nii_data

def gen_dMRI_fc1d_train_datasets(path, subject, ndwi, scheme, combine=None, whiten=True):
    """
    Generate fc1d training Datasets.
    path: the directory of the dataset folder (i.e the absolute path of Data-NODDI)
    subject: the subfolder name under Data-NODDI. (e.g. s01_still)
    ndwi: the first number of dwis to be read
    scheme: the rejection scheme to be used
    combine: if rejection shceme is presented, then True
    """
    # ltype are a list of label types
    ltype = ['NDI' , 'ODI', 'FWF']
    # ltype for Experiment1
    ltype_NDI = ['NDI']
    ltype_ODI = ['ODI']
    ltype_FWF = ['FWF'] 
    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/mask-e.nii datasets/mask/mask_' + subject + '.nii')      
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
        
    # load diffusion data
    data = load_nii_image(path + '/' + subject + '/diffusion.nii', mask)
    
    # Select the inputs.
    if combine is not None:
        data = data[..., combine == 1]
    else:
        data = data[..., :ndwi]

    # Whiten the data.
    if whiten:
        data = data / data.mean() - 1.0
    print(data.shape)

    # load labels, without scaling
    label = np.zeros((data.shape[0] , len(ltype)))
    for i in range(len(ltype)):
       filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
       temp = load_nii_image(filename,mask) 
       label[:, i] = temp.reshape(temp.shape[0]) 
     
    print(label.shape)

    # remove possible NAN values in parameter maps
    for i in range(label.shape[0]):
        if np.isnan(label[i]).any():
            label[i] = 0
            data[i] = 0

    # save datasets
    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'data':data})
    savemat('datasets/label/' + subject + '-' + '3label' +'-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'label':label})

"""
Utility functions to generate the patches (2D and 3D) for patched-CNN training
"""
def gen_2d_patches(data, mask, size, stride):
    """
    generate 2d patches
    """
    patches = []
    for layer in range(mask.shape[2]):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend = np.array([x, y]) + size
                lxend, lyend = np.array([x, y]) + stride
                if mask[x:lxend, y:lyend, layer].sum() > 0:
                    patches.append(data[x:xend, y:yend, layer, :])

    return np.array(patches)

def gen_3d_patches(data, mask, size, stride):
    """
    generate 3d patches
    """
    patches = []
    for layer in np.arange(0, mask.shape[2], stride):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend, layerend = np.array([x, y, layer]) + size
                lxend, lyend, llayerend = np.array([x, y, layer]) + stride
                if mask[x:lxend, y:lyend, layer:llayerend].sum() > 0:
                    if (layerend > mask.shape[2]):
                        return np.array(patches)
                    patch = data[x:xend, y:yend, layer: layerend, :]
                    patches.append(patch)

    # return np.array(patches)

def gen_dMRI_conv2d_train_datasets(path, subject, ndwi, scheme, patch_size, label_size, base=1, test=False, combine=None, whiten=True):
    """
    Generate Conv2D Dataset.
    """
    # ltype are a list of label types
    ltype = ['NDI' , 'ODI', 'FWF']
    # ltype for Experiment1
    ltype_NDI = ['NDI']
    ltype_ODI = ['ODI']
    ltype_FWF = ['FWF'] 
    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/mask-e.nii datasets/mask/mask_' + subject + '.nii')
    #load data, labels and mask
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    data = load_nii_image(path + '/' + subject + '/diffusion.nii')
    label = np.zeros(mask.shape + (len(ltype_NDI),))
    for i in range(len(ltype_NDI)):
        filename = path + '/' + subject + '/' + subject + '_' + ltype_NDI[i] + '.nii'
        label[:, :, :, i] = load_nii_image(filename)

    # select the inputs
    if combine is not None:
        data = data[..., combine == 1]
    else:
        data = data[..., :ndwi]
    
    if whiten:
        data = data / data.mean() - 1.0   

    # remove possible NAN values in parameter maps
    for i in range(label.shape[0]):
        if np.isnan(label[i]).any():
            label[i] = 0
            data[i] = 0 

    # generate patched datasets
    offset = base - (patch_size - label_size) / 2
    label = label[base:-base, base:-base, base:-base, :]
    # mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    mask = mask[base:-base, base:-base, base:-base]
    # data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat')['data']
    data = data[:, :, base:-base, :]

    if offset:
        data = data[offset:-offset, offset:-offset, :, :12]

    patches = gen_2d_patches(data, mask, patch_size, label_size)
    label = gen_2d_patches(label, mask, label_size, label_size)
    print(patches.shape)
    print(label.shape)

    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '2d.mat', {'data':patches})
    savemat('datasets/label/' + subject + 'NDI' +'-' + str(ndwi) + '-' + scheme + '-' + '2d.mat', {'label':label})

def gen_dMRI_conv3d_train_datasets(path, subject, ndwi, scheme, patch_size, label_size, base=1, test=False, combine=None, whiten=True):
    """
    Generate Conv3D Dataset.
    """
    # ltype are a list of label types
    ltype = ['NDI' , 'ODI', 'FWF']
    # ltype for Experiment1
    ltype_NDI = ['NDI']
    ltype_ODI = ['ODI']
    ltype_FWF = ['FWF'] 
    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/mask-e.nii datasets/mask/mask_' + subject + '.nii')
    #load data, labels and mask
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    data = load_nii_image(path + '/' + subject + '/diffusion.nii')
    label = np.zeros(mask.shape + (len(ltype_NDI),))
    for i in range(len(ltype_NDI)):
        filename = path + '/' + subject + '/' + subject + '_' + ltype_NDI[i] + '.nii'
        label[:, :, :, i] = load_nii_image(filename)

    # select the inputs
    if combine is not None:
        data = data[..., combine == 1]
    else:
        data = data[..., :ndwi]
    
    if whiten:
        data = data / data.mean() - 1.0

    # remove possible NAN values in parameter maps
    for i in range(label.shape[0]):
        if np.isnan(label[i]).any():
            label[i] = 0
            data[i] = 0    
    
    #generate patched datasets
    offset = base - (patch_size - label_size) / 2
    # labels = loadmat('datasets/label/' + subject + 'NDI' + '-' + str(ndwi) + '-' + scheme + '1d.mat')['label']
    label = label[base:-base, base:-base, base:-base, :]
    # mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    mask = mask[base:-base, base:-base, base:-base]
    # data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'data':data})
    data = data[:, :, base:-base, :]

    if offset:
        data = data[offset:-offset, offset:-offset, :, :12]

    patches = gen_3d_patches(data, mask, patch_size, label_size)
    patches = patches.reshape(patches.shape[0], -1)

    label = gen_3d_patches(label, mask, label_size, label_size)
    print(patches.shape)
    print(label.shape)

    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '3d.mat', {'data':patches})
    savemat('datasets/label/' + subject +'NDI' + '-' + str(ndwi) + '-' + scheme + '-' + '3d.mat', {'label':label})

def gen_dMRI_test_datasets(path, subject, ndwi, scheme, combine=None,  fdata=True, flabel=True, whiten=True):
    """
    Generate testing Datasets.
    """
    # ltype are a list of label types
    ltype = ['NDI' , 'ODI', 'FWF']
    # ltype for Experiment1
    ltype_NDI = ['NDI']
    ltype_ODI = ['ODI']
    ltype_FWF = ['FWF']
    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/mask-e.nii datasets/mask/mask_' + subject + '.nii')   
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
            
    if fdata:
        data = load_nii_image(path + '/' + subject + '/diffusion.nii')
        # data = load_nii_image(path + '/' + subject + '/diffusion.nii', mask)
        
        # Select the inputs.
        if combine is not None:
            data = data[..., combine == 1]
        else:
            data = data[..., :ndwi]

        # Whiten the data.
        if whiten:
            data = data / data[mask > 0].mean() - 1.0
            # data = data / data.mean() - 1.0
        
        print(data.shape)
        savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat', {'data':data})

    if flabel:
        label = np.zeros(mask.shape + (len(ltype_NDI),))
        for i in range(len(ltype_NDI)):
            filename = path + '/' + subject + '/' + subject + '_' + ltype_NDI[i] + '.nii'
            label[:, :, :, i] = load_nii_image(filename)
        print(label.shape)
        savemat('datasets/label/' + subject + 'NDI' + '-' + str(ndwi) + '-' + scheme + '.mat', {'label':label})

def fetch_train_data_MultiSubject(subjects, model, ndwi, scheme):
    """
    #Fetch train data.
    """
    data_s = None
    labels = None
    scheme = 'first'

    if model[:4] == 'fc1d':
        dim='1d.mat'
    if model[:6] == 'conv2d':
        dim='2d.mat'
    if model[:6] == 'conv3d':
        dim='3d.mat'

    for subject in subjects:
        label = loadmat('datasets/label/' + subject + 'NDI' + '-' + str(ndwi) + '-' + scheme + '-' + dim)['label']
        data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + dim)['data']

        if data_s is None:
            data_s = data
            labels = label
        else:
            data_s = np.concatenate((data_s, data), axis=0)
            labels = np.concatenate((labels, label), axis=0)

    data = np.array(data_s)
    label = np.array(labels)

    return data, label

def shuffle_data(data, label):
    """
    Shuffle data.
    """
    size = data.shape[-1]
    datatmp = np.concatenate((data, label), axis=-1)
    np.random.shuffle(datatmp)
    return datatmp[..., :size], datatmp[..., size:]

def repack_pred_label(pred, mask, model, ntype):
    """
    Get.
    """
    if model[7:13] == 'single':
        label = np.zeros(mask.shape + (1,))
    else:
        label = np.zeros(mask.shape + (ntype,))
    
    if model[:6] == 'conv2d':
        # label[1:-1, 1:-1, :, :] = pred.transpose(1, 2, 0, 3)
        label[1:-1, 1:-1, :, :] = pred
    elif model[:6] == 'conv3d':
        label[1:-1, 1:-1, 1:-1, :] = pred
    else:
        label = pred.reshape(label.shape)
    
    # label[:,:,:,1]=label[:,:,:,1]/1000 # scale MD back while saving nii
    return label