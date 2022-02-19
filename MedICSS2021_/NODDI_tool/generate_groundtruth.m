CreateROI('diffusion.nii', 'mask-e.nii', 'NODDI_roi.mat');
protocol = FSL2Protocol('bvals','bvecs');
noddi = MakeModel('WatsonSHStickTortIsoV_B0');
batch_fitting('NODDI_roi.mat', protocol, noddi, 'S04FittedParams.mat', 8)
SaveParamsAsNIfTI('S04FittedParams.mat', 'NODDI_roi.mat', 'mask-e.nii', 's04')