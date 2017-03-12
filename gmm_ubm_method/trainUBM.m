clc;
clear;
addpath('./netlab');
addpath('./voicebox');
% train UBM (Universal Background Model)
trainFile = 'testSpeech_003.wav';
[trainData, fs] = audioread(trainFile);
% vad
isSpeech = vadsohn(trainData, fs);
trainSpeech = trainData(isSpeech==1);
% extract mfcc feature
mfcc = melcepst(trainSpeech, fs);
% generate UBM model
component = 32;
gmm_ubm = gmm(size(mfcc, 2), component, 'diag');
options = zeros(1, 18);
options(3) = 1e-10;
options(14) = 20;
gmm_ubm = gmmem(gmm_ubm, mfcc, options);

save('gmm_ubm.mat', 'gmm_ubm');
