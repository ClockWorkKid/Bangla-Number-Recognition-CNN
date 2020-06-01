# Bangla Number Recognition with CNN on MATLAB

> This project detects Bengali numbers from voice data using a deep convolutional neural network model running on MATLAB.

> The project was ported from a MATLAB example project titled <a href="https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html">"Speech Command Recognition Using MATLAB"</a>

> Face recognition algorithm was used as a subsidiary part of the project following the MATLAB <a href="https://www.mathworks.com/help/vision/ref/vision.cascadeobjectdetector-system-object.html">Cascade Object Detector</a>

## Table of Contents

- [Project Overview](#overview)
- [Algorithm](#algorithm)
- [Data Collection](#data)
- [Training Model](#training)
- [Testing Model](#testing)
- [Discussion](#discussion)
- [References](#reference)
- [Team](#team)

## Project Overview

The project focuses on development of Bangla Speech Recognition, and as a subsection of the process, a convolutional neural network was trained to recognize bengali numbers from voice input. As a practical implementation of the number recogntion system, a simple interface has been designed targeting a digital library reception counter. Head on to the <a href="https://github.com/ClockWorkKid/Bangla-Number-Recognition-CNN/tree/master/Codes">"Codes"</a> directory for the complete working project. For an overview of the dataset generation and model training process, head to the <a href="https://github.com/ClockWorkKid/Bangla-Number-Recognition-CNN/tree/master/Dataset">"Dataset"</a> directory. The instructions can be followed to create your own dataset for training and testing the speech recognition algorithm.
(The project was built on MATLAB R2019a, compatibility has not been tested on previous versions of MATLAB.)

## Algorithm

The voice recognition algorithm works as follows:

- Voice is sampled at a rate of 16 KHz with a microphone, and a window is created with a size of 8192 samples (approximately 0.5 second sample window).
- Audio window is converted into a speech spectrogram and a spectrogram is generated. Created spectrogram is passed through a trained convolutional neural network model as image and classified as a number.
- New samples are passed through the network at runtime, and samples are overlapped for better prediction.

## Data Collection

- Voice data was collected from students for training the network. A simple microphone was used, and students were asked to iterate through numbers in different tones. After recording, post-processing was done via MATLAB to manually crop out audio corresponding to numbers (each clip of 8192 samples at 16 kHz), and sorted in classes. A total of approximately 2200 audio samples were generated in this way, amounting to 220 instances per class on average.
- The trimmed audio files were combined into a .mat file to feed into the training network as a single dataset.
- Additional noise data was sampled for training the network with a more robust performance.

## Training Model

- MATLAB Neural Network Toolbox was used to train the model. Due to the size of the dataset, the model trained can be assumed to be overfit. Training time requires approximately 30 minutes on a machine with I7-6500 CPU (not trained on GPU). 

## Testing Model

- After training model, it was tested with real voice input. As described before, voice data was sampled and sent to the classification network, and the predicted results are printed on screen. Despite being overfit during training, the runtime results are satisfactory. The model was able to handle noise as well.

## Discussion

Further work can be done for improving the network performance.
- Voice data has been used as is in the model, but several filtration techniques can be used to filter out additional noise.
- In place of spectrogram, MFCC can be used as a featureset for the recognition algorithm (using a different network model). Phoneme segmentation can be done instead of brute force spectrogram based detection. Phoneme segmentation can also work for natural language processing.
- Data used for training the model and in the end testing in real time have high correlation due to people participating in data collection, and environment. For more robust training, data has to be collected from different environments.

## References

- G.R. Bradski "Real Time Face and Object Tracking as a Component of a Perceptual User Interface", Proceedings of the 4th IEEE Workshop on Applications of Computer Vision, 1998.
- Viola, Paul A. and Jones, Michael J. "Rapid Object Detection using a Boosted Cascade of Simple Features", IEEE CVPR, 2001.
- Dalal, N. and B. Triggs. “Histograms of Oriented Gradients for Human Detection” Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, June 2005, pp. 886-893.
- Warden P. "Speech Commands: A public dataset for single-word speech recognition", 2017. Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz. Copyright Google 2017. The Speech Commands Dataset is licensed under the Creative Commons Attribution 4.0 license, available here: https://creativecommons.org/licenses/by/4.0/legalcode 

## Team

- Lead: Mir Sayeed Mohammad (Undergrad Student, BUET EEE)
- Co-lead: Md. Asif Iqbal (Undergrad Student, BUET EEE)
- Member: Dipon Paul (Undergrad Student, BUET EEE)
- Member: Sohan Mahmud (Undergrad Student, BUET EEE)
- Member: Azizul Zahid (Undergrad Student, BUET EEE)

- Supervisor: Dr. Celia Shahnaz (Professor, Dept of EEE, BUET)
- Supervisor: Sadman Sakib Ahbab Jarif (Lecturer, Dept of EEE, BUET)





