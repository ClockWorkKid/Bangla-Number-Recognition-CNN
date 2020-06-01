%{
Files:
1. 'recorder.m' : take the recording, and the data is saved
2. 'save_samples.m' : for cropping samples, follow the instructions 
3. 'sort_files.m' : for manually classifying the cropped samples
4. 'dataset_gen.m' : generate the dataset for training network model
5. 'train.m' : train the neural network model
6. 'command_recognition.m' : speech recognition from microphone

this program allows to fast sort the cropped data, first each data under
the entire set of cropped audios is loaded, played in matlab for
recognization, and then you have to specify which class the audio clip
belongs to. The program will then move the files to the corresponding data
folders

Cropped data is saved in the folder 'truncated'. Move the files from there
to 'Dataset_sorted\source\'

0-9 ==> bangla digit recognition class (give input '0'-'9')
noise ==> sample of noise (give input 10)
trash ==> rejected samples to be deleted (give input 11)
%}

clear all, close all, clc

path = 'Dataset_sorted';
path_from = 'source';
path_to = ["0","1","2","3","4","5","6","7","8","9","noise","trash"];

files = dir([path,'\',path_from,'\*.wav']);

L = length (files);

for idx = 1:L
    [audioData,Fs] = audioread([path,'\',path_from,'\',files(idx).name]);
    
    flag = 0;
    % play the sound in a loop until valid class is given as input
    
    while (flag==0)
        sound(audioData,Fs);
        data_class = input('What number is this?   ','s');
        for input_classes = 0:11
            if strcmp(int2str(input_classes),data_class)
                flag = 1;
                break;
            end
        end
        data_class = str2num(data_class);
    end
    
    movefile([path,'\',path_from,'\',files(idx).name],...
        [path,'\',char(path_to(data_class+1)),'\',files(idx).name],'f');
    
    clc
end
