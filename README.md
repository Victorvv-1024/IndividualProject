# IndividualProject
This repo is for the UCL undergraduate final year individual project

A. Guidance of packages installation and environment setting for DL-dMRI data analysis

1. Install Anaconda/Miniconda where you can use Conda commands to easily configure different environments. (Follow the installation here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Setting the environment for Python version of 3.7 

    # Create a new environment with python version of 3.7
    conda create --name your_env_name python=3.7   

    # Activate the environment when you need to use it
    conda activate your_env_name 
    
    # Deactivate the enviroment when you have finished using it   
    conda deactivate
 
3. Install necessary python packages
    
    conda activate your_env_name

    # Install tensorflow (which now includes keras) these two librarys are used for deep learning in python
    pip install tensorflow==2.3.1

    # Install scipy and numpy these libraries are used for performing mathmatical calculations on datasets 
    pip install scipy
    pip install numpy==1.17.0

    # Install nipy, this library is used for loading NIfTI images (.nii/.nii.gz). This is how MRI images are normally saved
    pip install nipy==0.4.2
    
4. Download MRIcron for convenient visualisation of Nifty files
    https://www.nitrc.org/projects/mricron
    https://www.nitrc.org/frs/?group_id=152

B. Related papers to read

1. The work that we are trying to replicate:
    https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13400

2. Basics about DTI:
    https://onlinelibrary.wiley.com/doi/epdf/10.1002/jmri.1076

3. To help you understand the general problem of DL in dMRI:
    https://ieeexplore.ieee.org/document/7448418
    
C. The codes and datasets
    
1. Datasets can be downloaded from this link (about 500 MB):
https://liveuclac-my.sharepoint.com/:u:/g/personal/ucacong_ucl_ac_uk/ET3o_rKS8s5EubrqHO7VvWEB6eVI_OvSaox78p7zpvntSw?e=cVQehj

Descriptions will be provided in Readme file
