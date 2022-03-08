"""
Functions for nii manipulation
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from nipy import load_image, save_image
from nipy.core.api import Image
import tensorflow as tf
import nibabel

def mask_nii_data(data, mask):
    """
    mask nii data from 3-D into 1-D
    """
    mask = mask.flatten()
    print('mask has shape: ' +str(mask.shape))
    data = data.reshape(mask.shape[0], -1)
    print('data befor masking')
    print(data.shape)

    # test the number of masked voxels is correct
    count = 0
    for i in range(data.shape[0]):
        if mask[i] > 0:
            count+=1
    print('masked count is: ' +str(count))

    data = data[mask > 0]
    print('data after masking')
    print(data.shape)
    print('the ratio of used voxels is: ' + str(data.shape[0]/mask.shape[0]))
    return data

def unmask_nii_data(data, mask):
    """
    unmask nii data from 1-D into 3-D
    """
    shape = mask.shape
    mask = mask.flatten()

    # Format the new data
    value = np.zeros(mask.shape)
    value[mask > 0] = data

    return value.reshape(shape)

def load_nii_image(filename, mask=None, patch=None):
    """
    Get data from nii image.
    """
    nii = load_image(filename)
    data = nii.get_data()
    
    if mask is not None:
        data = mask_nii_data(data, mask)

    return data

def save_nii_image(filename, data, template, mask=None):
    """
    Save data into nii image.
    """
    if mask is not None:
        data = unmask_nii_data(data, mask)

    tmp =load_image(template)
    img = Image(data, tmp.coordmap, tmp.metadata)
    try:
        save_image(img, filename)
    except IOError:
        path = '/'.join(filename.split('/')[:-1])
        os.system('mkdir -p ' + path)
        save_image(img, filename)

