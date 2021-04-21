%% Load Commands Data Set
% Use |audioDatastore| to create a datastore that contains the file
% names and the corresponding labels. Use the folder names as the label
% source. Specify the read method to read the entire audio file. Create a
% copy of the datastore for later use.

ads = audioDatastore("dataset", ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')
ads0 = copy(ads);

countEachLabel(ads)

%% Split Data into Training, Validation, and Test Sets

train_fraction = 0.7;
validation_fraction = 0.2;

total_data = uint32(length(ads.Labels));
train_index = uint32(total_data*train_fraction);
validation_index = uint32(total_data*validation_fraction);

ads = ads.shuffle();        % randomize order
adsTrain = ads.subset(1:train_index);
adsValidation = ads.subset(train_index+1:train_index+validation_index);
adsTest = ads.subset(train_index+validation_index+1:total_data);

disp("Dataset splitting complete");

%% Compute Speech Spectrograms
% To prepare the data for efficient training of a convolutional neural
% network, convert the speech waveforms to log-mel spectrograms.
%
% Define the parameters of the spectrogram calculation. |segmentDuration|
% is the duration of each speech clip (in seconds). |frameDuration| is the
% duration of each frame for spectrogram calculation. |hopDuration| is the
% time step between each column of the spectrogram. |numBands| is the
% number of log-mel filters and equals the height of each spectrogram.

segmentDuration = 0.5120;   %duration of each data (samples/Fs = 8192/16000)
frameDuration = 0.0128;     %duration of each frame in spectrogram calc
hopDuration = 0.00512;      %time step between each column of the spectrogram
numBands = 40;              %number of log-bark filters and equals the height of each spectrogram

% Compute the spectrograms for the training, validation, and test sets
% by using the supporting function
% <matlab:edit(fullfile(matlabroot,'examples','deeplearning_shared','main','speechSpectrograms.m'))
% |speechSpectrograms|>. The |speechSpectrograms| function uses
% |melSpectrogram| for the log-mel spectrogram calculations. To obtain data
% with a smoother distribution, take the logarithm of the spectrograms
% using a small offset |epsil|.
epsil = 1e-6;

XTrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
XTrain = log10(XTrain + epsil);

XValidation = speechSpectrograms(adsValidation,segmentDuration,frameDuration,hopDuration,numBands);
XValidation = log10(XValidation + epsil);

XTest = speechSpectrograms(adsTest,segmentDuration,frameDuration,hopDuration,numBands);
XTest = log10(XTest + epsil);

YTrain = adsTrain.Labels;
YValidation = adsValidation.Labels;
YTest = adsTest.Labels;

disp("Spectrograms computed");

%% Visualize Data
% Plot the waveforms and spectrograms of a few training examples. Play the
% corresponding audio clips.
specMin = min(XTrain(:));
specMax = max(XTrain(:));
idx = randperm(size(XTrain,4),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    
    subplot(2,3,i+3)
    spect = XTrain(:,:,1,idx(i));
    pcolor(spect)
    caxis([specMin+2 specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

% Training neural networks is easiest when the inputs to the network have a reasonably
% smooth distribution and are normalized. To check that the data distribution
% is smooth, plot a histogram of the pixel values of the training data.
figure
histogram(XTrain,'EdgeColor','none','Normalization','pdf')
axis tight
ax = gca;
ax.YScale = 'log';
xlabel("Input Pixel Value")
ylabel("Probability Density")


% Plot the distribution of the different class labels in the training and
% validation sets. The test set has a very similar distribution to the
% validation set.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")
subplot(2,1,2)
histogram(YValidation)
title("Validation Label Distribution")

disp("Data visualization complete")

%% Add Data Augmentation
% Create an augmented image datastore for automatic augmentation and
% resizing of the spectrograms. Translate the spectrogram randomly up to 10
% frames (100 ms) forwards or backwards in time, and scale the spectrograms
% along the time axis up or down by 20 percent. Augmenting the data can
% increase the effective size of the training data and help prevent the
% network from overfitting. The augmented image datastore creates augmented
% images in real time during training and inputs them to the network. No
% augmented spectrograms are saved in memory.
sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter( ...
    'RandXTranslation',[-10 10], ...
    'RandXScale',[0.8 1.2], ...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',augmenter);

disp("Data augmentation complete");

%% Define Neural Network Architecture
% Create a simple network architecture as an array of layers. Use
% convolutional and batch normalization layers, and downsample the feature
% maps "spatially" (that is, in time and frequency) using max pooling
% layers. Add a final max pooling layer that pools the input feature map
% globally over time. This enforces (approximate) time-translation
% invariance in the input spectrograms, allowing the network to perform the
% same classification independent of the exact position of the speech in
% time. Global pooling also significantly reduces the number of parameters
% in the final fully connected layer. To reduce the possibility of the
% network memorizing specific features of the training data, add a small
% amount of dropout to the input to the last fully connected layer.
%
% The network is small, as it has only five convolutional layers with few
% filters. |numF| controls the number of filters in the convolutional
% layers. To increase the accuracy of the network, try increasing the
% network depth by adding identical blocks of convolutional, batch
% normalization, and ReLU layers. You can also try increasing the number of
% convolutional filters by increasing |numF|.
%
% Use a weighted cross entropy classification loss.
% <matlab:edit(fullfile(matlabroot,'examples','deeplearning_shared','main','weightedClassificationLayer.m'))
% |weightedClassificationLayer(classWeights)|> creates a custom
% classification layer that calculates the cross entropy loss with
% observations weighted by |classWeights|. Specify the class weights in the
% same order as the classes appear in |categories(YTrain)|. To give each
% class equal total weight in the loss, use class weights that are
% inversely proportional to the number of training examples in each class.
% When using the Adam optimizer to train the network, the training
% algorithm is independent of the overall normalization of the class
% weights.
classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

timePoolSize = ceil(imageSize(2)/8);
dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
     
    maxPooling2dLayer([1 timePoolSize])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

disp("Model Created");

%% Train Network
% Specify the training options. Use the Adam optimizer with a mini-batch
% size of 128. Train for 25 epochs and reduce
% the learning rate by a factor of 10 after 20 epochs.
miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20);

disp("Training configuration complete");
%%
% Train the network. If you do not have a GPU, then training the network
% can take time. To load a pretrained network instead of training a
% network from scratch, set |doTraining| to |false|.

doTraining = false;
if doTraining
    disp("Model Training Started");
    trainedNet = trainNetwork(augimdsTrain,layers,options);
    save('\commandNet.mat', 'trainedNet');
    disp("Model Training Finished");
else
    load('commandNet.mat','trainedNet');
    disp("Model loaded");
end


%% Evaluate Trained Network
% Calculate the final accuracy of the network on the training set (without
% data augmentation) and validation set. The network is very accurate on
% this data set. However, the training, validation, and test data all have
% similar distributions that do not necessarily reflect real-world
% environments. This limitation particularly applies to the |unknown|
% category, which contains utterances of only a small number of words.
YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")


% Plot the confusion matrix. Display the precision and recall for each
% class by using column and row summaries. Sort the classes of the
% confusion matrix. The largest confusion is between unknown words and
% commands, _up_ and _off_, _down_ and _no_, and _go_ and _no_.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% When working on applications with constrained hardware resources such as
% mobile applications, consider the limitations on available memory and
% computational resources. Compute the total size of the network in
% kilobytes and test its prediction speed when using a CPU. The prediction
% time is the time for classifying a single input image. If you input
% multiple images to the network, these can be classified simultaneously,
% leading to shorter prediction times per image. When classifying streaming
% audio, however, the single-image prediction time is the most relevant.
info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")

for i=1:100
    x = randn(imageSize);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end
disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")

