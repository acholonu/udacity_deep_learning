# Overview

This document contains notes from the Udacity deep learning course.

## Python packages need to install

- [OpenCV](https://docs.opencv.org/4.5.5/index.html): `pip install opencv-python`.  This is a computer vision library that helps you do image processing and machine learning on images.  Great for Convolutional Nerual Networks (CNN's).
- [Pytorch](https://pytorch.org/get-started/locally/) `pip3 install torch torchvision`.  Deep learning framework.  A framework for create neural networks.  Also, has dummy dataset that you can use. Documentation can be found here <https://pytorch.org/docs/stable/index.html>

## Activation Functions

The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values.

A ReLU activation function stands for "Rectified Linear Unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the positive part of the input, x. So, for an input image with any negative pixel values, this would turn all those values to 0, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.

## Convolutional Neural Networks

