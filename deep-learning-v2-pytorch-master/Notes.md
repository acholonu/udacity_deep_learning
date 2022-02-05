# Overview

This document contains notes from the Udacity deep learning course. 

## Activation Functions

The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values.

A ReLU activation function stands for "Rectified Linear Unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the positive part of the input, x. So, for an input image with any negative pixel values, this would turn all those values to 0, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.