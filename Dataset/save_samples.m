%{
Files:
1. 'recorder.m' : take the recording, and the data is saved
2. 'save_samples.m' : for cropping samples, follow the instructions 
3. 'sort_files.m' : for manually classifying the cropped samples
4. 'dataset_gen.m' : generate the dataset for training network model
5. 'train.m' : train the neural network model
6. 'command_recognition.m' : speech recognition from microphone

Run this section of the code first, the sound file is loaded in the
workspace. Then open the variable in matlab signal analyzer app, and crop
off the required parts of the data. After selecting the required parts,
just export the data to the workspace, and then run the following section
%}

clear all; close all; clc

[sound,~] = audioread('Original Data\4.wav'); % specify file name here


%% 

%{
The original loaded sound track is first deleted from the workspace, then
using 'who' the workspace variables (exported from signal analyzer) are
listed, and we iterate through each data and save them with a specified
window length of 1024x8 = 8192 samples at a sampling rate of 16000. zeros
are added in case of small windows, or the data is truncated in case of
exceeding boundaries.

Cropped data is saved in the folder 'Cropped Data'. Move the files from there
to 'Dataset_sorted\source\'

The moving is done manually so that in any case of file copy/saving errors,
previouslly cropped data is not harmed.
%}

clear sound;
myvars = who;
Fs = 16000;
def_len = 8*1024;
roll = 203;

for idx=1:length(myvars)
    
    sample = eval(char(myvars(idx)));
    filename = ['Cropped Data\roll_',int2str(roll),'_samp_',int2str(idx),'.wav'];
    
    if length(sample) > def_len
        idx_0 = uint32((length(sample)-def_len)/2);
        resized_sample = sample(idx_0 : idx_0 + def_len - 1 , 1);
        
    elseif length(sample) < def_len
        resized_sample = zeros(def_len,1);
        idx_0 = uint32((def_len-length(sample))/2);
        resized_sample(idx_0 : idx_0 + length(sample) - 1 , 1) = sample;
    else
        resized_sample = sample;
    end
    
    audiowrite(filename,resized_sample, Fs);
    
end

disp('Cropped samples saved');

