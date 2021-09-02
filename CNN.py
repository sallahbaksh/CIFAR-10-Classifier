import numpy as np

from Layers import ReLuLayer, Conv, AvgPool, CNNFCL, Softmax


class CNN:
    def __init__(self):
        self.conv1 = Conv(nb_filters=6, filter_size=5, nb_channels=1)
        self.tanh1 = ReLuLayer(None)
        self.pool1 = AvgPool(filter_size=2, stride=2)
        self.conv2 = Conv(nb_filters=16, filter_size=5, nb_channels=6)
        self.tanh2 = ReLuLayer(None)
        self.pool2 = AvgPool(filter_size=2, stride=2)
        self.pool2_shape = None
        self.fc1 = CNNFCL(row=120, column=5 * 5 * 16)
        self.tanh3 = ReLuLayer(None)
        self.fc2 = CNNFCL(row=84, column=120)
        self.tanh4 = ReLuLayer(None)
        self.fc3 = CNNFCL(row=10, column=84)
        self.softmax = Softmax()

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forwardPropagate(self, X):
        # X is (BATCHx1x32x32)
        conv1 = self.conv1.forwardPropagate(X)  # This is (6x28x28)
        act1 = self.tanh1.forwardPropagate(conv1)
        pool1 = self.pool1.forwardPropagate(act1)  # This is (6x14x14)
        conv2 = self.conv2.forwardPropagate(pool1)  # (16x10x10)
        act2 = self.tanh2.forwardPropagate(conv2)
        pool2 = self.pool2.forwardPropagate(act2)  # This is (16x5x5)
        self.pool2_shape = pool2.shape
        pool2_flatten = pool2.reshape(self.pool2_shape[0], -1)  # This is (1x400)
        fc1 = self.fc1.forwardPropagate(pool2_flatten)  # This is (1x120)
        act3 = self.tanh3.forwardPropagate(fc1)
        fc2 = self.fc2.forwardPropagate(act3)  # This is (1x84)
        act4 = self.tanh4.forwardPropagate(fc2)
        fc3 = self.fc3.forwardPropagate(act4)  # This is (1x10)
        y_pred = self.softmax.forwardPropagate(fc3)
        return y_pred

    def backwardPropagate(self, y_pred, y):
        grad = self.softmax.backwardPropagate(y_pred, y)
        grad, dW5, db5, = self.fc3.backwardPropagate(grad)
        grad = self.tanh4.backwardPropagate(grad)
        grad, dW4, db4 = self.fc2.backwardPropagate(grad)
        grad = self.tanh3.backwardPropagate(grad)
        grad, dW3, db3 = self.fc1.backwardPropagate(grad)
        grad = grad.reshape(self.pool2_shape)
        grad = self.pool2.backwardPropagate(grad)
        grad = self.tanh2.backwardPropagate(grad)
        grad, dW2, db2 = self.conv2.backwardPropagate(grad)
        grad = self.pool1.backwardPropagate(grad)
        grad = self.tanh1.backwardPropagate(grad)
        grad, dW1, db1 = self.conv1.backwardPropagate(grad)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3,
            'dW4': dW4, 'db4': db4,
            'dW5': dW5, 'db5': db5
        }

        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['W' + str(i + 1)] = layer.W['val']
            params['b' + str(i + 1)] = layer.b['val']

        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W['val'] = params['W' + str(i + 1)]
            layer.b['val'] = params['b' + str(i + 1)]


class SGD:
    def __init__(self, eta, params):
        self.eta = eta
        self.params = params

    def update_params(self, grads):
        for key in self.params:
            self.params[key] = self.params[key] - self.eta * grads['d' + key]
        return self.params


class AdamGD:

    def __init__(self, eta, rho1, rho2, epsilon, params):
        self.eta = eta
        self.beta1 = rho1
        self.beta2 = rho2
        self.epsilon = epsilon
        self.params = params

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):
        for key in self.params:
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['d' + key]
            self.rmsprop['sd' + key] = (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (
                    grads['d' + key] ** 2)
            self.params[key] = self.params[key] - (self.eta * self.momentum['vd' + key]) / (
                    np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)

        return self.params
