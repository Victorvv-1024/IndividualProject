"""
Read from nii data and given under-samping scheme and save .mat for network use
Check /utils/data_utils.py for the defined functions for data formatting process

Usage:
1. To generate a base voxel-wise training dataset:
    
    python3 FormatData.py --path $DataDir --subjects s01_still --label_type label --base 
  
2. To genrate a voxel-wise training dataset for ANN:
    python3 FormatData.py --path $DataDir --subjects s01_still --label_type label --fc1d

3. To genrate a voxel-wise training dataset for 2D CNN:
    python3 FormatData.py --path $DataDir --subjects s01_still --label_type label --conv2d

4. To genrate a voxel-wise training dataset for 3D CNN:
    python3 FormatData.py --path $DataDir --subjects s01_still --label_type label --conv3d
"""
import argparse
from unicodedata import name
import numpy as np
from sympy import arg

from utils import gen_base_datasets, gen_conv2d_datasets, gen_conv3d_datasets, gen_fc1d_datasets



def parser():
    """
    Create a parser
    """
    # parser for genegrate the dataset 
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", help="The path of data folder", default='/home/vw/Desktop/IndividualProject/MedICSS2021_/Data-NODDI')
    parser.add_argument("--subjects", help="subjects ID", nargs='*', default='s01_still')
    parser.add_argument("--nDWI", help="The number of volumes", type=int, default=96)
    parser.add_argument("--scheme", help="The sampling scheme used")
    parser.add_argument("--base", help="generate base data for testing", action="store_true")
    parser.add_argument("--fc1d", help="generate fc1d data for training", action="store_true")
    parser.add_argument("--conv2d", help="generate 2d patches for training", action="store_true")
    parser.add_argument("--conv3d", help="generate 3d patches for training", action="store_true")
    parser.add_argument("--patch_size", metavar='ksize', help="Size of the kernels", type=int, default=3)
    parser.add_argument("--label_size", help="Size of the label", type=int, default=1)
    parser.add_argument("--label_type", help="generate data with acquired label. N for NDI, O for ODI and F for FWF; A all labels"
                        ,choices=['N', 'O', 'F', 'A'], nargs=1)

    return parser

def generate_data(args):
    # Training parameter
    path = args.path
    subjects = args.subjects
    base = args.base
    fc1d = args.fc1d
    conv2d = args.conv2d
    conv3d = args.conv3d

    label_type = args.label_type
    patch_size = args.patch_size
    label_size = args.label_size
    test = args.test
    Nolabel = args.Nolabel

    # determine the input volumes using the first n volumes
    nDWI = args.nDWI
    scheme = "first"

    # determin the input volumes using a scheme file
    combine = None
    schemefile = args.scheme
    if schemefile is not None:
        combine = np.loadtxt('schemes/' + schemefile)
        combine = combine.astype(int)
        nDWI = combine.sum()
        scheme = schemefile

    # Format the dataset
    if base:
        for subject in subjects:
            if Nolabel:
                gen_base_datasets(path, subject, label_type=label_type, fdata=True, flabel=False)
            else: 
                gen_base_datasets(path, subject, label_type=label_type, fdata=True, flabel=True)
    if fc1d:
        for subject in subjects:
            gen_fc1d_datasets(path, subject, patch_size, label_size, label_type, base=1, test=False)
    if conv2d:
        for subject in subjects:
            gen_conv2d_datasets(path, subject, patch_size, label_size, label_type, base=1, test=False)

    if conv3d:
        for subject in subjects:
            gen_conv3d_datasets(path, subject, patch_size, label_size, label_type, base=1, test=False)

if __name__ == '__main__':
    args = parser().parse_args()
    generate_data(args)

