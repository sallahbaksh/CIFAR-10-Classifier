#!/bin/python3

import concurrent.futures as cf
import os
import pickle

import numpy as np
from skimage import transform


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


class Conv:
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        np.random.seed(0)
        self.num_filters = nb_filters
        self.filter_size = filter_size
        self.num_channels = nb_channels
        self.stride = stride
        self.padding = padding

        # attempt xavier init
        self.W = {
            'val': np.random.randn(self.num_filters, self.num_channels, self.filter_size, self.filter_size) * np.sqrt(
                1. / self.filter_size),
            'grad': np.zeros((self.num_filters, self.num_channels, self.filter_size, self.filter_size))}
        self.b = {'val': np.random.randn(self.num_filters) * np.sqrt(1. / self.num_filters),
                  'grad': np.zeros(self.num_filters)}

        self.input_layer = None

    def forwardPropagate(self, X):
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.num_filters
        n_H = int((n_H_prev + 2 * self.padding - self.filter_size) / self.stride) + 1
        n_W = int((n_W_prev + 2 * self.padding - self.filter_size) / self.stride) + 1

        X_col = im2col(X, self.filter_size, self.filter_size, self.stride, self.padding)
        w_col = self.W['val'].reshape((self.num_filters, -1))
        b_col = self.b['val'].reshape(-1, 1)
        out = w_col @ X_col + b_col
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.input_layer = X, X_col, w_col
        return out

    def backwardPropagate(self, grad_in):
        X, X_col, w_col = self.input_layer
        m, _, _, _ = X.shape
        self.b['grad'] = np.sum(grad_in, axis=(0, 2, 3))
        grad_in = grad_in.reshape(grad_in.shape[0] * grad_in.shape[1], grad_in.shape[2] * grad_in.shape[3])
        grad_in = np.array(np.vsplit(grad_in, m))
        grad_in = np.concatenate(grad_in, axis=-1)
        dX_col = w_col.T @ grad_in
        dw_col = grad_in @ X_col.T
        dX = col2im(dX_col, X.shape, self.filter_size, self.filter_size, self.stride, self.padding)
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.num_channels, self.filter_size, self.filter_size))

        return dX, self.W['grad'], self.b['grad']


class Softmax:
    def __init__(self):
        pass

    def forwardPropagate(self, X):
        e_x = np.exp(X - np.max(X))
        return e_x / np.sum(e_x, axis=1)[:, np.newaxis]

    def backwardPropagate(self, y_pred, y):
        return y_pred - y


# Helper functions found at https://hackmd.io/@bouteille/B1Cmns09I
def one_hot_encoding(y):
    N = y.shape[0]
    Z = np.zeros((N, 10))
    Z[np.arange(N), y] = 1
    return Z


def save_params_to_file(model, optimizer_choice):
    terminal_path = ["./save_weights/"]
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath is None:
        raise FileNotFoundError(
            "save_params_to_file(): Impossible to find save_weights/ from current folder. You need to manually add "
            "the path to it in the \'terminal_path\' list and the run the function again.")

    weights = model.get_params()
    with open(dirPath + "final_weights.pkl" + ("ADAM" if optimizer_choice else ""), "wb") as f:
        pickle.dump(weights, f)


def load_params_from_file(model, optimizer_choice):
    terminal_path = ["./save_weights/final_weights.pkl" + ("ADAM" if optimizer_choice else "")]

    filePath = None
    for path in terminal_path:
        if os.path.isfile(path):
            filePath = path
    if filePath is None:
        raise FileNotFoundError(
            'load_params_from_file(): Cannot find final_weights.pkl from your current folder. You need to '
            'manually add it to terminal_path list and the run the function again.')

    pickle_in = open(filePath, 'rb')
    params = pickle.load(pickle_in)
    model.set_params(params)
    return model


def dataloader(X, y, BATCH_SIZE):
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t + BATCH_SIZE, ...], y[t:t + BATCH_SIZE, ...]


def resize_dataset(dataset):
    args = [dataset[i:i + 1000] for i in range(0, len(dataset), 1000)]

    def f(chunk):
        return transform.resize(chunk, (chunk.shape[0], 1, 32, 32))

    with cf.ThreadPoolExecutor() as executor:
        res = executor.map(f, args)

    res = np.array([*res])
    res = res.reshape(-1, 1, 32, 32)
    return res


class AvgPool:
    def __init__(self, filter_size, stride=1, padding=0):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_layer = None

    def forwardPropagate(self, X):
        self.input_layer = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.padding - self.filter_size) / self.stride) + 1
        n_W = int((n_W_prev + 2 * self.padding - self.filter_size) / self.stride) + 1

        X_col = im2col(X, self.filter_size, self.filter_size, self.stride, self.padding)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        A_pool = np.mean(X_col, axis=1)
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    def backwardPropagate(self, in_grad):
        X = self.input_layer
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        d_flatten = in_grad.reshape(n_C, -1) / (self.filter_size * self.filter_size)
        dX_col = np.repeat(d_flatten, self.filter_size * self.filter_size, axis=0)
        dX = col2im(dX_col, X.shape, self.filter_size, self.filter_size, self.stride, self.padding)
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX


class CNNFCL:
    def __init__(self, row, column):
        self.row = row
        self.col = column

        # attempt xavier
        self.W = {'val': np.random.randn(self.row, self.col) * np.sqrt(1. / self.col), 'grad': 0}
        self.b = {'val': np.random.randn(1, self.row) * np.sqrt(1. / self.row), 'grad': 0}

        self.input_layer = None

    def forwardPropagate(self, fc):
        self.input_layer = fc
        A_fc = np.dot(fc, self.W['val'].T) + self.b['val']
        return A_fc

    def backwardPropagate(self, grad):
        fc = self.input_layer
        m = fc.shape[0]

        self.W['grad'] = (1 / m) * np.dot(grad.T, fc)
        self.b['grad'] = (1 / m) * np.sum(grad, axis=0)

        return np.dot(grad, self.W['val']), self.W['grad'], self.b['grad']


# Code inspired and adapted from
# https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood
# \-226523ce7fbf#:~:text=Simply%20put%2C%20im2col%20is%20a,result%20after%20reshaping%20the%20output.
def get_indices(X_shape, HF, WF, stride, pad):
    m, n_C, n_H, n_W = X_shape
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
    level1 = np.repeat(np.arange(HF), WF)
    level1 = np.tile(level1, n_C)
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    slide1 = np.tile(np.arange(WF), HF)
    slide1 = np.tile(slide1, n_C)
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
