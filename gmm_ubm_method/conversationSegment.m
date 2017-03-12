clc;
clear;
addpath('./voicebox');
addpath('./netlab');
% load gmm speaker model and gmm UBM model
load('speakerModel.mat');
load('gmm_ubm.mat');
component = 32;
% % % % % do speaker recognition block by block
conversationFile = 'testSpeech_003.wav';
[conversationData, fs] = audioread(conversationFile);
isSpeech = vadsohn(conversationData, fs);
blockLen = 1000 * fs / 1000;
shiftLen = 500 * fs / 1000;
blockNum = floor(size(conversationData,1) / shiftLen) - 1;

threshld = 0.1;
SegmentP = zeros(blockNum);
segment = [];
count = 1;
startIndex = -shiftLen + 1;
for i = 1:blockNum
    startIndex = startIndex + shiftLen;
    endIndex = startIndex + blockLen - 1;
    blockSpeech = conversationData(startIndex:endIndex);
    if sum(isSpeech(startIndex:endIndex))/blockLen < 0.5 % not speech
        segment = [segment; blockSpeech];
       continue; 
    end    
    % feature extraction
    mfcc = melcepst(blockSpeech, fs);
    % compute similarity
    topmix = gmmprob_index(gmm_ubm, mfcc, component);
    prob_ubm = gmmprob_ntop(gmm_ubm, topmix, mfcc);
    prob_speaker = gmmprob_ntop(gmm_speaker, topmix, mfcc);
    SegmentP(i) = prob_speaker - prob_ubm;
    result = SegmentP(i) > threshld;
    if i == 1
       flag = result; 
    end
    % do segment
    if flag ~= result
        if flag == true
            filename = ['./result/' num2str(count) '_true.wav'];
        else
            filename = ['./result/' num2str(count) '_false.wav'];
        end
        audiowrite(filename, segment, fs);
        segment = blockSpeech(1:shiftLen);
        count = count + 1;
        flag = result;
    else
        segment = [segment;blockSpeech(1:shiftLen)];
    end
end

if size(segment) ~= 0
    if flag == true
         filename = ['./result/' num2str(count) '_true.wav'];
    else
         filename = ['./result/' num2str(count) '_false.wav'];
    end
    audiowrite(filename, segment, fs);
end

figure;
plot(1:blockNum, SegmentP);