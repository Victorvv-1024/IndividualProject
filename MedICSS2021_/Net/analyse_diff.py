"""
 Read the diff map and analyse it
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


path = '../Net/nii/'
# read the difference map
ndi_diff = nib.load(path+'s01_still-12-fc1d-patch_1-base_1-layer_4-label_NDI_difference_synthetic.nii').get_fdata()
odi_diff = nib.load(path+'s01_still-12-fc1d-patch_1-base_1-layer_4-label_ODI_difference_synthetic.nii').get_fdata()
fwf_diff = nib.load(path+'s01_still-12-fc1d-patch_1-base_1-layer_4-label_FWF_difference_synthetic.nii').get_fdata()

# flatten
ndi_diff = np.array(ndi_diff.tolist()).flatten()
odi_diff = np.array(odi_diff.tolist()).flatten()
fwf_diff = np.array(fwf_diff.tolist()).flatten()

diff = [ndi_diff, odi_diff, fwf_diff]

# print off some statistics
print('ndi diff mean is: ' + str(np.mean(ndi_diff)) + ' and the std is: ' + str(np.std(ndi_diff)))
print('odi diff mean is: ' + str(np.mean(odi_diff)) + ' and the std is: ' + str(np.std(odi_diff)))
print('fwf diff mean is: ' + str(np.mean(fwf_diff)) + ' and the std is: ' + str(np.std(fwf_diff)))

# plot the histogram
b, bins, patches = plt.hist(fwf_diff, 100,color='red')
plt.xlim([-0.05,0.05])
plt.ylim([0,50000])
plt.show()