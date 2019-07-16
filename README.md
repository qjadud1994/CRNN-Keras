# CRNN (CNN+RNN) 

OCR(Optical Character Recognition) consists of text localization + text recognition.
(text localization finds where the characters are, and text recognition reads the letters.)

You can use this [text localizaion model](https://github.com/qjadud1994/OCR_Detector) I have studied.

After performing localization, each text area is cropped and used as input for text recognition.
An example of text recognition is typically the CRNN

Combining the [text detector](https://github.com/qjadud1994/OCR_Detector) with a [CRNN](https://github.com/qjadud1994/CRNN-Keras) makes it possible to create an OCR engine that operates end-to-end.

## CRNN

**[CRNN](https://arxiv.org/pdf/1507.05717.pdf)** is a network that combines CNN and RNN to process images containing sequence information such as letters.

It is mainly used for OCR technology and has the following advantages.
1. End-to-end learning is possible.
2. Sequence data of arbitrary length can be processed because of LSTM which is free in size of input and output sequence.
3. There is no need for a detector or cropping technique to find each character one by one.

You can use CRNN for OCR, license plate recognition, text recognition, and so on. 
It depends on what data you are training.

I used a slightly modified version of the original CRNN model.
(Input size : 100x30 -> 128x64 & more CNN Layer)

## Network

![CRNN Network](https://github.com/qjadud1994/CRNN-Keras/blob/master/photo/Network.jpg)

### Convolutional Layer
Extracts features through CNN Layer (VGGNet, ResNet ...).

### Recurrent Layer
Splits the features into a certain size and inserts them into the input of the Bidirectional LSTM or GRU.

### Transcription Layer

Conversion of Feature-specific predictions to Label using CTC (Connectionist Temporal Classification).

- - -

## license plate recognition using CRNN

I used CRNN to recognize license plates in Korea.

![Type of license plate](https://github.com/qjadud1994/CRNN-Keras/blob/master/photo/license%20plate.jpg)

I learned the following kinds of Korean license plates.

I updated the [Korean License Plate Synthetic image generator](https://github.com/qjadud1994/Korean-license-plate-Generator.git) for those who lacked license plate pictures.


## Result
![Result](https://github.com/qjadud1994/CRNN-Keras/blob/master/photo/result.jpg)

CRNN works well for license plate recognition as follows.


## How to Training

First, you need a lot of cropped license plate images. <br/>
And in my case I expressed the number of the license plate with the image file name. <br/>
(The license plate number 1234 is indicated as "1234.jpg"). <br/>
(You can also define labeling with txt or csv files if you want. [(ex)0001.jpg "1234" \n 0002.jpg "0000" ...)

Since I used Korean license plates, I expressed the Korean language on the license plate in English.

![Example](https://github.com/qjadud1994/CRNN-Keras/blob/master/DB/train/A18sk6897.jpg)
<br/>
(example) A18sk6897 <br/>
A : 서울 <br/>
sk : 나 <br/>

After creating training data in this way, put it in 'DB/train' directory and run [training.py](https://github.com/qjadud1994/CRNN-Keras/blob/master/training.py).

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
