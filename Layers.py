#!/bin/python3

import numpy as np


class InputLayer:
    def __init__(self, input_layer):
        self.mean = np.mean(input_layer, axis=0)
        self.std = np.std(input_layer, axis=0, ddof=1)

    def forwardPropagate(self, input_layer):
        self.std[self.std == 0] = 1
        return (input_layer - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__


class FullyConnectedLayer:
    def __init__(self, sizein, sizeout, eta):
        np.random.seed(0)
        lower, upper = -(np.sqrt(6.0) / np.sqrt(sizein + sizeout)), (np.sqrt(6.0) / np.sqrt(sizein + sizeout))
        self.weights = 0.0001 * (np.random.rand(sizein, sizeout) - 0.5)
        self.biases = 0.0001 * (np.random.rand(1, sizeout) - 0.5)
        self.input_layer = None
        self.eta = eta
        self.epochs = 0
        self.sw = 0
        self.rw = 0
        self.sb = 0
        self.rb = 0

    def forwardPropagate(self, dataIn):
        self.input_layer = dataIn
        return (dataIn @ self.weights) + self.biases

    def backwardPropagate(self, gradient):
        p1, p2, delta = 0.9, 0.999, 10 ** -8
        self.epochs += 1
        gradout = gradient @ self.gradient()
        deltaJ = (self.input_layer.T @ gradient) / (self.input_layer.shape[0])
        self.sw = p1 * self.sw + (1 - p1) * deltaJ
        self.rw = p2 * self.rw + (1 - p2) * (deltaJ ** 2)

        deltaJ = (np.ones((self.input_layer.shape[0], 1))).T @ gradient / self.input_layer.shape[0]
        self.sb = p1 * self.sb + (1 - p1) * deltaJ
        self.rb = p2 * self.rb + (1 - p2) * (deltaJ ** 2)

        self.weights -= self.eta * np.divide((self.sw / (1 - p1 ** self.epochs)),
                                             np.sqrt(self.rw / (1 - p2 ** self.epochs)) + delta)
        self.biases -= self.eta * np.divide((self.sb / (1 - p1 ** self.epochs)),
                                            np.sqrt(self.rb / (1 - p2 ** self.epochs)) + delta)
        return gradout

    def simpleBackwardPropagate(self, gradient, learning_rate: float):
        xx = learning_rate / gradient.shape[0]
        self.weights -= xx * (self.input_layer.T @ gradient)
        self.biases -= xx * (np.ones((1, gradient.shape[0])) @ gradient)
        return gradient @ self.gradient()

    def gradient(self):
        return self.weights.T

    def __repr__(self):
        return self.__class__.__name__


class ReLuLayer:
    def __init__(self, inp):
        self.input_layer = inp

    def forwardPropagate(self, input_layer):
        self.input_layer = input_layer
        return np.maximum(0, input_layer)

    def backwardPropagate(self, in_grad):
        return np.multiply(in_grad, self.gradient())

    def gradient(self):
        return np.where(self.input_layer > 0, 1, 0)

    def __repr__(self):
        return self.__class__.__name__


class SigmoidLayer:
    def __init__(self, inp):
        self.input_layer = inp

    def forwardPropagate(self, input_layer):
        self.input_layer = input_layer
        return 1 / (1 + np.exp(-input_layer))

    def backwardPropagate(self, in_grad):
        return np.multiply(in_grad, self.gradient())

    def gradient(self):
        g = self.forwardPropagate(self.input_layer)
        return g * (1 - g)

    def __repr__(self):
        return self.__class__.__name__


class SoftmaxLayer:
    def __init__(self, inp):
        self.input_layer = inp

    def forwardPropagate(self, input_layer):
        self.input_layer = input_layer
        ediff = np.exp(input_layer - np.amax(input_layer))
        return np.nan_to_num(np.divide(ediff.T, np.sum(ediff, axis=1) + 10 ** -7).T, nan=1)

    def backwardPropagate(self, in_grad):
        return np.multiply(in_grad, self.gradient())

    def gradient(self):
        g = self.forwardPropagate(self.input_layer)
        return g * (1 - g)

    def __repr__(self):
        return self.__class__.__name__


class TanhLayer:
    def __init__(self, inp):
        self.input_layer = inp

    def forwardPropagate(self, input_layer):
        self.input_layer = input_layer
        return np.tanh(input_layer)

    def backwardPropagate(self, in_grad):
        return np.multiply(in_grad, self.gradient())

    def gradient(self):
        g = self.forwardPropagate(self.input_layer)
        return 1 - g * g

    def __repr__(self):
        return self.__class__.__name__


class DropoutLayer:
    def __init__(self, inp, p=0.5):
        self.input_layer = inp
        self.p = p
        self.mask = None

    def forwardPropagate(self, input_layer):
        self.input_layer = input_layer
        self.mask = np.random.binomial(1, self.p, size=input_layer.shape) / self.p
        out = np.multiply(input_layer, self.mask)
        return out.reshape(input_layer.shape)

    def backwardPropagate(self, in_grad):
        return np.multiply(in_grad, self.gradient())

    def gradient(self):
        return self.mask

    def __repr__(self):
        return self.__class__.__name__
