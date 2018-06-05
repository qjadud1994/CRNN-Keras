# CRNN (CNN+RNN) 

**CRNN** is a network that combines CNN and RNN to process images containing sequence information such as letters.

https://arxiv.org/pdf/1507.05717.pdf

It is mainly used for OCR technology and has the following advantages.
1. End-to-end learning is possible.
2. Sequence data of arbitrary length can be processed because of LSTM which is free in size of input and output sequence.
3. There is no need for a detector or cropping technique to find each character one by one.

You can use CRNN for OCR, license plate recognition, text recognition, and so on. 
It depends on what data you are training.

## Network

![CRNN Network](https://github.com/qjadud1994/CRNN-Keras/blob/master/photo/Network.jpg)

### Convolutional Layer
Extracts features through CNN Layer (VGGNet, ResNet ...).

### Recurrent Layer
Splits the features into a certain size and inserts them into the input of the Bidirectional LSTM or GRU.

### Transcription Layer

Conversion of Feature-specific predictions to Label using CTC (Connectionist Temporal Classification).



## license plate recognition using CRNN

I used CRNN to recognize license plates in Korea.

![Type of license plate](https://github.com/qjadud1994/CRNN-Keras/blob/master/photo/license%20plate.jpg)

I learned the following kinds of Korean license plates.



### Result
![Result](https://github.com/qjadud1994/CRNN-Keras/blob/master/photo/result.jpg)

CRNN works well for license plate recognition as follows.




## File Description

os : Ubuntu 16.04.4 LTS
GPU : GeForce GTX 1080 (8GB)
Python : 3.5.2
Tensorflow : 1.5.0
Keras : 2.1.3
CUDA, CUDNN : 9.0, 7.0

|       File         |Description                                       |
|--------------------|--------------------------------------------------|
|Model .py           |Network using CNN (VGG) + Bidirectional LSTM      |
|Model_GRU. py       |Network using CNN (VGG) + Bidirectional GRU       |
|Image_Generator. py |Image batch generator for training                |
|parameter. py       |Parameters used in CRNN                           |
|training. py        |CRNN training                                     |
|Prediction. py      |CRNN prediction                                   |
