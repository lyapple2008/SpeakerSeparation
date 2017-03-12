clc;
clear;
addpath('./netlab');
addpath('./voicebox');
% train speaker model
trainFile = 'Speaker_zhu.wav';
[trainData, fs] = audioread(trainFile);
% vad
isSpeech = vadsohn(trainData, fs);
trainSpeech = trainData(isSpeech==1);
% extract mfcc feature
mfcc = melcepst(trainSpeech, fs);
% generate speaker model
load('gmm_ubm.mat');
component = 32;
gmm_speaker = gmmmap(gmm_ubm, mfcc, component);
save('speakerModel.mat', 'gmm_speaker');