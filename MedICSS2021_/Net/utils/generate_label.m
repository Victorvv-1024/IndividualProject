%% Set up the dependencies
% set the NODDI toolbox root and nifti1 matlab
NODDItoolboxRootDir = '/home/vw/Desktop/IndividualProject/MedICSS2021_/NODDI_tool';
nifti1RootDir = '/home/vw/Desktop/IndividualProject/MedICSS2021_/nifti_matlab';

% update the path
addpath(genpath(NODDItoolboxRootDir));
addpath(genpath(nifti1RootDir));


% set the NODDI dataset as root
NODDIDataRootDir = '/home/vw/Desktop/IndividualProject/MedICSS2021_/Data-NODDI/s04_still_reg';

%% load the mask and the NODDI_roi
%
NODDI_roi = load([NODDIDataRootDir '/NODDI_roi.mat']);
synthetic_roi = load([NODDIDataRootDir '/roi_synthetic.mat']);
mask = NODDI_roi.mask;
idx = NODDI_roi.idx;
roi = synthetic_roi.noisy_vals_T;
% save the new roi file
filepath = NODDIDataRootDir;
name = 'NODDI_roi_synthetic';
extension = '.mat';
matname = fullfile(filepath, [name extension]);
save(matname, "mask", "roi", "idx");

%% Get the imaging protocol
bvalFile = [NODDIDataRootDir '/bvals'];
bvecFile = [NODDIDataRootDir '/bvecs'];

protocol = FSL2Protocol(bvalFile, bvecFile);


%% Create the NODDI model structure  
noddi = MakeModel('WatsonSHStickTortIsoV_B0');


%% Run the NODDI fitting with the function batch_fitting
batch_fitting('NODDI_roi_synthetic.mat', protocol, noddi, 'S04FittedParams_synthetic.mat', 8);

%% Convert the estimated NODDI parameters into volumetric parameter maps
SaveParamsAsNIfTI('S04FittedParams_synthetic.mat', 'NODDI_roi_synthetic.mat', 'mask-e.nii', 'synthetic')
