"""
Functions for nii manipulation
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from nipy import load_image, save_image
from nipy.core.api import Image
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

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

def filter_mask(subpath, fwfpath, threshold=0.99):
    """
    By looking at the imgs generated, we have found out there are some regions that should not be included. Since they have values higher than 1.0
    And we have found out voxels have NDI and ODI values, while that voxel has GROUND TRUTH FWF 1.0
    This should indicate that that voxel should not even be included in the training
    Therefore we want to filter the each subject's mask first, by using their corresponding GROUND TRUTH FWF

    Args:
        subpath (string): the path of the subject folder
        fwfpath (string): the path of the corresponding fwf file
        threshold (float): the thresholds to be used to filter of the mask,
                           a stringnent threshold would be 0.9, the least stringnent threshold is 1.0
                           by default, it is set to 0.99
    """
    # fetch the mask data
    img_mask = nib.load(subpath+'mask-e.nii')
    original_mask = img_mask.get_fdata()
    original_affine = img_mask.affine
    shape = original_mask.shape # retain the shape of the mask
    origin_nonzeros = np.count_nonzero(original_mask)
    print('original mask has: ' + str(origin_nonzeros) + ' of nonzero voxels')
    # fetch the FWF data
    fwf = nib.load(fwfpath).get_fdata()
    # filter
    mask = original_mask.flatten() # this makes a copy of the orginal mask
    fwf = fwf.reshape(mask.shape[0]) # reshape fwf to the corresponding shape
    for i in range(len(mask)):
        # if fwf has high value, means there is no tissue
        # therefore, the voxel should be excluded
        if fwf[i] >= threshold:
            mask[i] = 0.0
    # reshape mask back
    mask = mask.reshape(shape)
    filter_nonzeros = np.count_nonzero(mask)
    print('filtered mask has: ' +str(filter_nonzeros) + ' of nonzero voxels')
    # save the mask
    filter_img = nib.Nifti1Image(mask, original_affine)
    nib.save(filter_img, subpath+'filtered_mask.nii')

def show_slices(slices, grayscale=True):
    """
    Function to display the slices

    Args:
        slices (list): a list of 2d ndarray that contains the data to be displayed
        grayscale (bool, optional): True, if diplay grayscale img. Defaults to True.
    """    
    fig, axes = plt.subplots(1, len(slices), figsize=(10,10))
    cax = fig.add_axes([0, 0, .3, .3])
    for i, slice in enumerate(slices):
        # use grayscale for displaying ref and pred imgs:
        if grayscale:
            cmap = plt.get_cmap('gray')
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            bounds = np.arange(0, 1.0, .01)
            idx = np.searchsorted(bounds, 0)
            bounds = np.insert(bounds, idx, 0)
            norm = BoundaryNorm(bounds, cmap.N)
            im = axes[i].imshow(slice.T, cmap=cmap, origin="lower", interpolation='none', norm=norm)
        else:
            # define the colormap
            cmap = plt.get_cmap('bwr')
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            # define the bins and normalize and forcing 0 to be part of the colorbar
            # define the min and max to be -1 and +1 respectively
            bounds = np.arange(-0.5, 0.5, .01)
            idx = np.searchsorted(bounds, 0)
            bounds = np.insert(bounds, idx, 0)
            norm = BoundaryNorm(bounds, cmap.N)
            im = axes[i].imshow(slice.T, cmap=cmap, origin="lower", interpolation='none', norm=norm)
        fig.colorbar(im, cax=cax, orientation='vertical')