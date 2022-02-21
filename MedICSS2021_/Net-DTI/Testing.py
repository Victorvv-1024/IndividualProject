"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Testing.py -h`

"""

from unicodedata import decimal
import numpy as np
import os
import time

from scipy.io import savemat, loadmat

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, parser, load_nii_image, unmask_nii_data, loss_funcs, fetch_train_data_MultiSubject


# Get parameter from command-line input
args = parser().parse_args()

train_subjects = args.train_subjects
test_subjects = args.test_subjects[0]
nDWI = args.DWI
scheme = args.scheme
mtype = args.model

lr = args.lr
epochs = args.epoch
kernels = args.kernels
layer = args.layer

loss = args.loss
batch_size = args.batch
patch_size = args.patch_size
label_size = patch_size - 2
base = args.base

# Constants
types = ['NDI' , 'ODI', 'FWF']
ntypes = len(types)
decay = 0.1

# Parameter name definition
savename = str(nDWI)+ '-'  + scheme + '-' + args.model + '-' + str(layer) + 'layer'

# Define the adam optimizer
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Load testing data
mask = load_nii_image('datasets/mask/mask_' + test_subjects + '.nii')
tdata = loadmat('datasets/data/' + test_subjects + '-' + str(nDWI) + '-' + scheme + '.mat')['data']
# print(tdata)

test_shape = args.test_shape
if test_shape is None:
  test_shape = tdata.shape[1:4]

# Define the model
model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape)
model.model(adam, loss_func, patch_size)
model.load_weight(savename)

weights = model._model.layers[1].get_weights()

# Predict on the test data.
pred = model.predict(tdata)
# Evluate on the test data
tlabel = loadmat('datasets/label/' + test_subjects + 'NDI' + '-' + str(nDWI) + '-' + scheme + '.mat')['label']
rmse = np.sqrt(np.mean((pred-tlabel)**2))
print(np.around(rmse,decimals=5))

# pred = repack_pred_label(pred, mask, mtype, ntypes_0)

# For experiment 2
# Save estimated measures to /nii folder as nii image
# os.system("mkdir -p nii")

# for i in range(ntypes_0):
#     data = pred[..., i]
#     filename = 'nii/' + test_subjects + '-' + type_0[i] + '-' + savename + '.nii'

#     data[mask == 0] = 0
#     save_nii_image(filename, data, 'datasets/mask/mask_' + test_subjects + '.nii', None)
