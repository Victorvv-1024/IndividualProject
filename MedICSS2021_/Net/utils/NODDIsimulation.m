%% Simulate NODDI data from parameter estimates

%
% author: Gary Zhang (gary.zhang@ucl.ac.uk)
%

%% Set up the dependencies

% set the NODDI toolbox root
NODDItoolboxRootDir = '/home/vw/Desktop/IndividualProject/NODDI/NODDI_toolbox_v1.05';

% update the path
addpath(genpath(NODDItoolboxRootDir));

% set the NODDI dataset as root
NODDIDataRootDir = '/home/vw/Desktop/IndividualProject/MedICSS2021_/Data-NODDI/s04_still_reg';

%% Load fitted parameters of NODDI example dataset

load([NODDIDataRootDir '/S04FittedParams.mat']);

%% Get the imaging protocol
% 
bvalFile = [NODDIDataRootDir '/bvals'];
bvecFile = [NODDIDataRootDir '/bvecs'];

protocol = FSL2Protocol(bvalFile, bvecFile);

%% test the first functions

% get the signal free vals for all voxels
% fix the random seed for reproducibility
rng(1);
% set SNR to the typical value of 30
SNR = 30;
% set b0 to the mean b=0 value in white matter - the value comes from
% inspecting the mlps output with a histogram
b0_mean = 600;

% work out the sigma of the noise distribution
sigma = b0_mean/SNR;

% work out the model name
modelName4NODDI = model.name;

% get the output from the function and transpose it
[noise_free_vals, noisy_vals] = noise(mlps,modelName4NODDI,protocol,sigma);

noisy_vals_T = transpose(noisy_vals);

%% save it to a directory
filepath = NODDIDataRootDir;
name = 'S04FittedParams_synthetic';
extension = '.mat';
matname = fullfile(filepath, [name extension]);
save(matname, "noisy_vals_T")
%% A function that works for every voxel
function [noise_free_vals, noisy_vals] = noise(a,modelName4NODDI,protocol,sigma)
    noise_free_vals = [];
    noisy_vals = [];
    for i = 1:size(a,1)
        NODDIparameters = a(i,:);
        % apply the correct scaling factors
        scale = GetScalingFactors(modelName4NODDI);
        NODDIparameters(1:(length(scale)-1)) = NODDIparameters(1:(length(scale)-1))./scale(1:(end-1));
        % work out the fibre orientation from the tissue parameters
        fibredir = GetFibreOrientation(modelName4NODDI, NODDIparameters);
        
        % set the constants variable to 0 - indicating it's not needed
        constants = 0;
        
        % compute the noise-free signal
        signalNoiseFree = SynthMeas(modelName4NODDI, NODDIparameters, protocol, fibredir, 0);

        % add to the noise_free_vals
        noise_free_vals = [noise_free_vals signalNoiseFree];
        
        % work out the number of noises required
        numOfMeasurements = length(signalNoiseFree);
        
        % allocate the memory for synthesised noises
        noiseRealChannel = zeros(numOfMeasurements, 1);
        noiseImagChannel = zeros(numOfMeasurements, 1);
        
        % allocate the memory for the resulting noisy measurements
        signalNoisy = zeros(numOfMeasurements, 1);
        
        % synthesise the dual-channel noise
        noiseRealChannel(:,1) = randn(numOfMeasurements, 1)*sigma;
        noiseImagChannel(:,1) = randn(numOfMeasurements, 1)*sigma;
        
        % add noise to the signal
        signalNoisy(:,1) = sqrt((signalNoiseFree + noiseRealChannel(:,1)).^2 + noiseImagChannel(:,1).^2);

        % add it to noisy vals
        noisy_vals = [noisy_vals signalNoisy];
    end
end
