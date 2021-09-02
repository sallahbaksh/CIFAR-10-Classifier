import os
import pickle

import numpy as np
import concurrent.futures as cf

from matplotlib import pyplot as plt
from skimage import transform


def plot_example(X, y, y_pred=None):
    """
        Plots 9 examples and their associate labels.

        Parameters:
        -X: Training examples.
        -y: true labels.
        -y_pred: predicted labels.
    """
    # Create figure with 3 x 3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    X, y = X[:9, 0, ...], y[:9]

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(X[i])

        # Show true and predicted classes.
        if y_pred is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], y_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots in a single Notebook cell.
    plt.show()


def plot_example_errors(X, y, y_pred):
    """
        Plots 9 example errors and their associate true/predicted labels.

        Parameters:
        -X: Training examples.
        -y: true labels.
        -y_pred: predicted labels.

    """
    incorrect = (y != y_pred)

    X = X[incorrect]
    y = y[incorrect]
    y_pred = y_pred[incorrect]

    # Plot the first 9 images.
    plot_example(X, y, y_pred)


def one_hot_encoding(y):
    """
        Performs one-hot-encoding on y.

        Parameters:
        - y: ground truth labels.
    """
    N = y.shape[0]
    Z = np.zeros((N, 10))
    Z[np.arange(N), y] = 1
    return Z


def save_params_to_file(model):
    """
        Saves model parameters to a file.
        Parameters:
        -model: a CNN architecture.
    """
    # Make save_weights/ accessible from every folders.
    terminal_path = ["./save_weights/"]
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError(
            "save_params_to_file(): Impossible to find save_weights/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    weights = model.get_params()
    if dirPath == '../fast/save_weights/':  # We run the code from demo notebook.
        with open(dirPath + "demo_weights1.pkl", "wb") as f:
            pickle.dump(weights, f)
    else:
        with open(dirPath + "final_weights1.pkl", "wb") as f:
            pickle.dump(weights, f)


def load_params_from_file(model, isNotebook=False):
    """
        Loads model parameters from a file.
        Parameters:
        -model: a CNN architecture.
    """
    if isNotebook:  # We run from demo-notebooks/
        pickle_in = open("../fast/save_weights/demo_weights.pkl1", 'rb')
        params = pickle.load(pickle_in)
        model.set_params(params)
    else:
        # Make final_weights.pkl file accessible from every folders.
        terminal_path = ["./save_weights/final_weights1.pkl"]

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
    """
        Returns a data generator.
        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t + BATCH_SIZE, ...], y[t:t + BATCH_SIZE, ...]


def resize_dataset(dataset):
    """
        Resizes dataset of MNIST images to (32, 32).
        Parameters:
        -dataset: a numpy array of size [?, 1, 28, 28].
    """
    args = [dataset[i:i + 1000] for i in range(0, len(dataset), 1000)]

    def f(chunk):
        return transform.resize(chunk, (chunk.shape[0], 1, 32, 32))

    with cf.ThreadPoolExecutor() as executor:
        res = executor.map(f, args)

    res = np.array([*res])
    res = res.reshape(-1, 1, 32, 32)
    return res


def train_val_split(X, y, val=40000):
    """
        Splits X and y into training and validation set.
        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    X_train, X_val = X[:val, :], X[val:, :]
    y_train, y_val = y[:val, :], y[val:, :]

    return X_train, y_train, X_val, y_val


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.
        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]


class Conv:
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {'val': np.random.randn(self.n_F, self.n_C, self.f, self.f) * np.sqrt(1. / self.f),
                  'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
        self.b = {'val': np.random.randn(self.n_F) * np.sqrt(1. / self.n_F), 'grad': np.zeros(self.n_F)}

        self.cache = None

    def forward(self, X):
        """
            Performs a forward convolution.

            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - out: previous layer convolved.
        """
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W['val'].reshape((self.n_F, -1))
        b_col = self.b['val'].reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - dout: error from previous layer.

            Returns:
            - dX: error of the current convolutional layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        # Compute bias gradient.
        self.b['grad'] = np.sum(dout, axis=(0, 2, 3))
        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        # Reshape dw_col into dw.
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))

        return dX, self.W['grad'], self.b['grad']


class AvgPool():

    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        """
            Apply average pooling.
            Parameters:
            - X: Output of activation function.

            Returns:
            - A_pool: X after average pooling layer.
        """
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        A_pool = np.mean(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    def backward(self, dout):
        """
            Distributes error through pooling layer.
            Parameters:
            - dout: Previous layer with the error.

            Returns:
            - dX: Conv layer updated with error.
        """
        X = self.cache
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        dout_flatten = dout.reshape(n_C, -1) / (self.f * self.f)
        dX_col = np.repeat(dout_flatten, self.f * self.f, axis=0)
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        # Reshape dX properly.
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX


class Fc():

    def __init__(self, row, column):
        self.row = row
        self.col = column

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {'val': np.random.randn(self.row, self.col) * np.sqrt(1. / self.col), 'grad': 0}
        self.b = {'val': np.random.randn(
            1, self.col) * np.sqrt(1. / self.col), 'grad': 0}

        self.cache = None

    def forward(self, fc):
        """
            Performs a forward propagation between 2 fully connected layers.
            Parameters:
            - fc: fully connected layer.

            Returns:
            - A_fc: new fully connected layer.
        """
        self.cache = fc
        A_fc = fc @ self.W['val'] + self.b['val']
        return A_fc

    def backward(self, deltaL):
        """
            Returns the error of the current layer and compute gradients.
            Parameters:
            - deltaL: error at last layer.

            Returns:
            - new_deltaL: error at current layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        fc = self.cache
        m = fc.shape[0]

        # Compute gradient.
        self.W['grad'] = (1 / m) * (fc.T @ deltaL)
        self.b['grad'] = (1 / m) * np.sum(deltaL, axis=0)

        # Compute error.
        new_deltaL = deltaL@ self.W['val'].T
        # We still need to multiply new_deltaL by the derivative of the activation
        # function which is done in TanH.backward().
        return new_deltaL, self.W['grad'], self.b['grad']


class SGD():

    def __init__(self, lr, params):
        self.lr = lr
        self.params = params

    def update_params(self, grads):
        for key in self.params:
            self.params[key] = self.params[key] - self.lr * grads['d' + key]
        return self.params


class AdamGD():

    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):

        for key in self.params:
            # Momentum update.
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['d' + key]
            # RMSprop update.
            self.rmsprop['sd' + key] = (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (
                    grads['d' + key] ** 2)
            # Update parameters.
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (
                    np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)

        return self.params


class TanH():

    def __init__(self, alpha=1.7159):
        self.alpha = alpha
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.
            Parameters:
            - X: input tensor.
        """
        self.cache = X
        return self.alpha * np.tanh(X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * (1 - np.tanh(X) ** 2)


class ReLU:

    def __init__(self):
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.
            Parameters:
            - X: input tensor.
        """
        self.cache = X
        return np.maximum(0, X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * np.where(X > 0, 1, 0)


class Softmax():

    def __init__(self):
        pass

    def forward(self, X):
        """
            Compute softmax values for each sets of scores in X.
            Parameters:
            - X: input vector.
        """
        e_x = np.exp(X - np.max(X))
        return e_x / np.sum(e_x, axis=1)[:, np.newaxis]

    def backward(self, y_pred, y):
        return y_pred - y


class Sigmoid():
    def __init__(self):
        pass

    def forward(self, input_layer):
        self.input_layer = input_layer
        return 1 / (1 + np.exp(-input_layer))

    def backward(self, in_grad):
        return np.multiply(in_grad, self.gradient())

    def gradient(self):
        g = self.forward(self.input_layer)
        return g * (1 - g)


class CrossEntropyLoss():

    def __init__(self):
        pass

    def get(self, y_pred, y):
        """
            Return the negative log likelihood and the error at the last layer.

            Parameters:
            - y_pred: model predictions.
            - y: ground truth labels.
        """
        loss = -np.sum(y * np.log(y_pred))
        return loss


class LogLoss:
    """The Log Loss Objective Class requires that the input to the eval function are values between 0 and 1.
    """

    def __init__(self):
        pass

    def get(self,y, yhat):
        return np.average(-1 * ((y * np.log(yhat + 10**-7)) + ((1 - y) * np.log(1 - yhat + 10**-7))))

    def gradient(self, yhat):
        return -1 * (self.y - yhat) / (yhat * (1 - yhat) + 10**-7)

    def forwardPropagate(self, input_layer):
        self.forward_prop = input_layer
        return self.forward_prop

    def __repr__(self):
        return self.__class__.__name__

class LogisticLoss():

    def __init__(self) -> None:
        pass

    def get(self, y_pred):
        return np.average(-np.log(y_pred + 10**-7))
