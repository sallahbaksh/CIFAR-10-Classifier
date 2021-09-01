import numpy as np

from Layers import *
from ObjectiveFuncs import *
from tqdm import trange


def one_hot_encoder(data):
    return np.squeeze(np.eye(np.max(data) + 1)[data.reshape(-1)])


def one_hot_decoder(x):
    return np.argmax(x, axis=1)


class MLP:
    def __init__(self, train_X, train_Y, test_X, test_Y, eta, epochs):
        self.train_Y = one_hot_encoder(train_Y)
        self.test_Y = one_hot_encoder(test_Y)
        self.test_X = test_X
        self.train_X = train_X
        self.eta = eta
        self.epochs = epochs
        self.arch = []
        self.inp_train = None
        self.inp_test = None
        self.train_y_hat = None
        self.test_y_hat = None

    def create_architecture(self, architecture: str):
        """
        architecture: A comma separated string, with the input layer, hidden layers and the objective layer;
                      The fully connected layer should display an input layer output layer (ex: FullyConnected 25)
        """
        self.arch = list(map(self.layerFactory, list(map(str.strip, architecture.split(',')))))
        self.inp_train = self.arch[0].forwardPropagate(self.train_X)
        self.inp_test = self.arch[0].forwardPropagate(self.test_X)
        fp = self.inp_train
        for layer in self.arch[1:]:
            fp = layer.forwardPropagate(fp)

    def train(self):
        for _ in trange(self.epochs):
            fp = self.inp_test
            for layer in self.arch[1:-1]:
                fp = layer.forwardPropagate(fp)
            self.test_y_hat = fp

            fp = self.inp_train
            for layer in self.arch[1:-1]:
                fp = layer.forwardPropagate(fp)
            self.train_y_hat = fp

            fp = self.arch[-1].gradient(self.train_y_hat)
            for layer in self.arch[-2:0: -1]:
                fp = layer.backwardPropagate(fp)

    def layerFactory(self, class_str):
        layer = {
            'Input': InputLayer(self.train_X),
            'Sigmoid': SigmoidLayer(None),
            'ReLu': ReLuLayer(None),
            'Softmax': SoftmaxLayer(None),
            'Tanh': TanhLayer(None),
            'CrossEntropy': CrossEntropy(self.train_Y),
            'LogLoss': LogLoss(self.train_Y),
            'LeastSquares': LeastSquares(self.train_Y),
        }

        if (v := class_str.split())[0] == 'FullyConnected':
            return FullyConnectedLayer(int(v[1]), int(v[2]), self.eta)
        elif (v := class_str.split())[0] == 'DropOut':
            return DropoutLayer(None, float(v[1]))
        elif class_str in layer:
            return layer[class_str]
        else:
            raise AttributeError

    def calculate_Accuracies(self):
        total_train = np.sum(np.equal(tyh := one_hot_decoder(self.train_y_hat), one_hot_decoder(self.train_Y)))
        total_test = np.sum(np.equal(vyh := one_hot_decoder(self.test_y_hat), one_hot_decoder(self.test_Y)))
        print(f'Train Accuracy: {total_train * 100 / tyh.shape[0]}%')
        print(f'Validation Accuracy: {total_test * 100 / vyh.shape[0]}%')
