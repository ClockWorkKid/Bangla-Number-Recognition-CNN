%% Detect Commands Using Streaming Audio from Microphone
% Test your newly trained command detection network on streaming audio from
% your microphone. If you have not trained a network, then type
% |load('commandNet.mat')| at the command line to load a pretrained network
% and the parameters required to classify live, streaming audio.

load('commandNet.mat','trainedNet');
disp("Model loaded");

%% 
segmentDuration = 0.5120;   %duration of each data (samples/Fs = 8192/16000)
frameDuration = 0.0128;     %duration of each frame in spectrogram calc
hopDuration = 0.00512;      %time step between each column of the spectrogram
numBands = 40;              %number of log-bark filters and equals the height of each spectrogram
epsil = 1e-6;

% Specify the audio sampling rate and classification rate in Hz and create
% an audio device reader that can read audio from your microphone.
fs = 16e3;
classificationRate = 20;
audioIn = audioDeviceReader('SampleRate',fs, ...
    'SamplesPerFrame',floor(fs/classificationRate));


% Specify parameters for the streaming spectrogram computations and
% initialize a buffer for the audio. Extract the classification labels of
% the network. Initialize buffers of half a second for the labels and
% classification probabilities of the streaming audio. Use these buffers to
% compare the classification results over a longer period of time and by
% that build 'agreement' over when a command is detected.
frameLength = floor(frameDuration*fs);
hopLength = floor(hopDuration*fs);
waveBuffer = zeros([8192,1]);

labels = trainedNet.Layers(end).Classes;
YBuffer(1:classificationRate/2) = categorical("background");
probBuffer = zeros([numel(labels),classificationRate/2]);

%%
% Create a figure and detect commands as long as the created figure exists.
% To stop the live detection, simply close the figure. 
h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

while ishandle(h)
    
    % Extract audio samples from the audio device and add the samples to
    % the buffer.
    x = audioIn();
    waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
    waveBuffer(end-numel(x)+1:end) = x;
    
    % Compute the spectrogram of the latest audio samples.
    spec = melSpectrogram(waveBuffer,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength - hopLength, ...
        'FFTLength',512, ...
        'NumBands',numBands, ...
        'FrequencyRange',[50,7000]);
    spec = log10(spec + epsil);
    spec = spec(:, 1:98);   % forcing size to match input of model
    
    % Classify the current spectrogram, save the label to the label buffer,
    % and save the predicted probabilities to the probability buffer.
    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','cpu');
    YBuffer(1:end-1)= YBuffer(2:end);
    YBuffer(end) = YPredicted;
    probBuffer(:,1:end-1) = probBuffer(:,2:end);
    probBuffer(:,end) = probs';
    
    % Plot the current waveform and spectrogram.
    subplot(2,1,1);
    plot(waveBuffer)
    axis tight
    ylim([-0.2,0.2])
    
    subplot(2,1,2)
    pcolor(spec)
    shading flat
    
    % Now do the actual command detection by performing a very simple
    % thresholding operation. Declare a detection and display it in the
    % figure title if all of the following hold:
    % 1) The most common label is not |background|.
    % 2) At least |countThreshold| of the latest frame labels agree.
    % 3) The maximum predicted probability of the predicted label is at
    % least |probThreshold|. Otherwise, do not declare a detection.
    [YMode,count] = mode(YBuffer);
    countThreshold = ceil(classificationRate*0.2);
    maxProb = max(probBuffer(labels == YMode,:));
    probThreshold = 0.7;
    subplot(2,1,1);
    if YMode == "background" || count<countThreshold || maxProb < probThreshold
        title(" ")
    else
        title(string(YMode),'FontSize',20)
    end
    
    drawnow
    
end