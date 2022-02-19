"""
Functions for Generating or save dataset.
"""

import os
import numpy as np
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
    label = np.zeros((data.shape[0] , len(ltype_FWF)))
    for i in range(len(ltype_FWF)):
       filename = path + '/' + subject + '/' + subject + '_' + ltype_FWF[i] + '.nii'
       temp = load_nii_image(filename,mask) 
       label[:, i] = temp.reshape(temp.shape[0]) 

    # load labels, if scaling is necessary 
    # filename = path + '/' + subject + '/' + subject + '_' + ltype[0] + '.nii'
    # temp = load_nii_image(filename,mask)
    # label[:, 0] = temp.reshape(temp.shape[0]) 

    # filename = path + '/' + subject + '/' + subject + '_' + ltype[1] + '.nii'
    # temp = load_nii_image(filename,mask) * 1000   # scale MD to the value around 1
    # label[:, 1] = temp.reshape(temp.shape[0]) 
     
    print(label.shape)

    # remove possible NAN values in parameter maps
    for i in range(label.shape[0]):
        if np.isnan(label[i]).any():
            label[i] = 0
            data[i] = 0

    # save datasets
    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'data':data})
    savemat('datasets/label/' + subject + 'FWF' +'-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'label':label})

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
        # data = load_nii_image(path + '/' + subject + '/diffusion.nii')
        data = load_nii_image(path + '/' + subject + '/diffusion.nii', mask)
        
        # Select the inputs.
        if combine is not None:
            data = data[..., combine == 1]
        else:
            data = data[..., :ndwi]

        # Whiten the data.
        if whiten:
            # data = data / data[mask > 0].mean() - 1.0
            data = data / data.mean() - 1.0
        
        print(data.shape)
        savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat', {'data':data})

    if flabel:
        # label = np.zeros(mask.shape + (len(ltype_NDI),))
        label = np.zeros((data.shape[0] , len(ltype_FWF)))
        for i in range(len(ltype_FWF)):
            # filename = path + '/' + subject + '/' + subject + '_' + ltype_NDI[i] + '.nii'
            # label[:, :, :, i] = load_nii_image(filename)
            filename = path + '/' + subject + '/' + subject + '_' + ltype_FWF[i] + '.nii'
            print(filename)
            temp = load_nii_image(filename,mask) 
            label[:, i] = temp.reshape(temp.shape[0])
        print(label.shape)
        savemat('datasets/label/' + subject + 'FWF' + '-' + str(ndwi) + '-' + scheme + '.mat', {'label':label})

def fetch_train_data_MultiSubject(subjects, ndwi, scheme):
    """
    #Fetch train data.
    """
    data_s = None
    labels = None

    for subject in subjects:
        label = loadmat('datasets/label/' + subject + 'FWF' +'-' + str(ndwi) + '-' + scheme + '-' + '1d.mat')['label']
        data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat')['data']

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
        label[1:-1, 1:-1, :, :] = pred.transpose(1, 2, 0, 3)
    elif model[:6] == 'conv3d':
        label[1:-1, 1:-1, 1:-1, :] = pred
    else:
        label = pred.reshape(label.shape)
    
    # label[:,:,:,1]=label[:,:,:,1]/1000 # scale MD back while saving nii
    return label