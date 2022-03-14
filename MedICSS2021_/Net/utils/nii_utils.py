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
    This is a function that turns the 3D data into 1D

    Args:
        data (ndarray): a 4D numpy array contains the value of the data. 
                        The first 3 dims suggest the width, height, slice of the DWI images,
                        the last dim suggests the input size (the volumes of DWI)
        mask (ndarray): a 3D numpy array contains the value of the mask

    Returns:
        ndarray: a 2D numpy array, the first dim suggests the value of masked voxels
                 the second dim suggests the input size
    """    
    # flatten the mask
    mask = mask.flatten()
    print('mask has shape: ' +str(mask.shape))
    # flatten the data, because the input layer for fc1d is define as (ndwi,)
    data = data.reshape(mask.shape[0], -1) 
    print('data befor masking has shape: ' +str(data.shape))
    data = data[mask > 0] # return the masked voxel, the masked voxel has postive values i.e. 1
    print('data after masking has shape: ' + str(data.shape) + ' the ratio of masked voxel is: ' + str(data.shape[0]/mask.shape[0]))
    return data

def unmask_nii_data(data, mask):
    """
    Unmask the 1D data into 3D

    Args:
        data (ndarray): a 2D array, where the first dim suggests the value of masked voxels,
                        the second dim suggests the input size
        mask (ndarray): a 3D array, the contains the value of the mask

    Returns:
        ndarray: a 4D array, the first 3 dims suggest the width, height, slice of the DWI images,
                the last dim suggests the input size (the volumes of DWI)
    """
    shape = mask.shape
    mask = mask.flatten()

    # Format the new data
    value = np.zeros(mask.shape)
    # the masked region has the same value as data, other regions are padded with 0
    value[mask > 0] = data 

    return value.reshape(shape)

def load_nii_image(filename, mask=None, patch=None):
    """
    Get data from nii image
    Args:
        filename (string): the abs path of the nii image
        mask (ndarray, optional): a 3D array that contains the value of the mask. Defaults to None.
        patch (int, optional): the size of the patch. Defaults to None.

    Returns:
        ndarray: an array that contains the value of the data
    """    """
    Get data from nii image.
    """
    nii = load_image(filename)
    data = nii.get_data()
    
    # if masking, then run mask_nii_data
    if mask is not None:
        data = mask_nii_data(data, mask)

    return data

def save_nii_image(filename, data, template, mask=None):
    """
    save the nii image
    Args:
        filename (string): the destination name of the saved image
        data (ndarray): the array contains the value of the data
        template (string): the path of the template image (i.e the abs of the mask)
        mask (ndarray, optional): the array contains the value of the mask. Defaults to None.
    """    """
    Save data into nii image.
    """
    # if masking
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

