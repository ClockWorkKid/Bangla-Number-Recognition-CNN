%{
Files:
1. 'recorder.m' : take the recording, and the data is saved
2. 'save_samples.m' : for cropping samples, follow the instructions
3. 'sort_files.m' : for manually classifying the cropped samples
4. 'dataset_gen.m' : generate the dataset for training network model
5. 'train.m' : train the neural network model
6. 'command_recognition.m' : speech recognition from microphone

Just run dataset_gen.m to generate the dataset
variables:
'audio_data' ==> (total_samples x 8192) double matrix of audioread() data
'labels' ==> (total_samples) categorical label column vector
'Fs' ==> sampling rate of the dataset
'total_samples' ==> number of total_samples in dataset
%}

clear all, close all, clc

directory = 'Dataset_sorted';
sub_dir = ["0","1","2","3","4","5","6","7","8","9","noise"];
dataset_name = 'Bangla_digit_voice_dataset_with_noise.mat';

len = 0;

% find the total available data samples
for idx = 1:11
    path = [directory,'\',char(sub_dir(idx))];
    files = dir([path,'\*.wav']);
    len = len + length (files);
end

disp('Initializing dataset generator');

audio_data = zeros(len,8192);           % blank dataset
labels = categorical(zeros(len,1));       % blank dataset labels
total_samples = 0;                  % total samples included in the dataset

% add data to dataset
for idx = 1:11
    
    path = [directory,'\',char(sub_dir(idx))];
    files = dir([path,'\*.wav']);
    
    L = length (files);
    
    for sample_NO = 1:L              % number of samples in current label
        
        total_samples = total_samples + 1;
        [audio_sample,Fs] = audioread([path,'\',files(sample_NO).name]);
        audio_data(total_samples,:) = audio_sample';
        labels(total_samples) = categorical(sub_dir(idx));
        
    end
end

% save dataset as .mat file
save(dataset_name,'audio_data','labels','Fs','total_samples');
disp('Dataset generation complete');
