% speechSpectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands)
% computes speech spectrograms for the files in the datastore ads.
% segmentDuration is the total duration of the speech clips (in seconds),
% frameDuration the duration of each spectrogram frame, hopDuration the
% time shift between each spectrogram frame, and numBands the number of
% frequency bands.

function X = mySpeechSpectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands,fs,numFiles)

disp("Computing speech spectrograms...");

numHops = ceil((segmentDuration - frameDuration)/hopDuration);
X = zeros([numBands,numHops,1,numFiles],'single');

for i = 1:numFiles
    
    x = ads(i,:)';
    frameLength = round(frameDuration*fs);
    hopLength = round(hopDuration*fs);
    
    spec = myAuditorySpectrogram(x,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength - hopLength, ...
        'NumBands',numBands, ...
        'Range',[50,7000], ...
        'WindowType','Hann', ...
        'WarpType','Bark', ...
        'SumExponent',2);
    
    % If the spectrogram is less wide than numHops, then put spectrogram in
    % the middle of X.
    w = size(spec,2);
    left = floor((numHops-w)/2)+1;
    ind = left:left+w-1;
    X(:,ind,1,i) = spec;
    
    if mod(i,100) == 0
        disp("Processed " + i + " files out of " + numFiles)
    end
    
end

disp("...done");

end