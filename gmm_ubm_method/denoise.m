clc;
clear;
% denoise testing file and train file
logmmse('Speaker_zhu.wav', 'Speaker_zhu_denoise.wav');
logmmse('testSpeech_003.wav', 'testSpeech_003_denoise.wav');
