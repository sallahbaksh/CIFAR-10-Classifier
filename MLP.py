import numpy as np

from Layers import *
from ObjectiveFuncs import *
from tqdm import trange
from sklearn.metrics import accuracy_score
learning_rate, epochs = 10 ** (-2), 2_000


def one_hot_encoder(data):
    return np.squeeze(np.eye(np.max(data) + 1)[data.reshape(-1)])


def one_hot_decoder(x):
    return np.argmax(x, axis=1)


class MLP:
    def __init__(self, tsx, tsy, vsx, vsy):
        self.train_y = one_hot_encoder(tsy)
        self.test_y = one_hot_encoder(vsy)
        self.test_x = vsx
        self.train_x = tsx
        self.arch = []
        self.inp_train = None
        self.inp_test = None
        self.train_y_hat = None
        self.test_y_hat = None

    def create_architecture(self, architecture: str):
        """architecture: A comma separated string, with the input layer, hidden layers and the objective layer;
                         The fully connected layer should display an input layer output layer (ex: FullyConnected 25)
        """
        self.arch = list(map(self.layerFactory, list(map(str.strip, architecture.split(',')))))
        # [print(x) for x in self.arch]
        self.inp_train = self.arch[0].forwardPropagate(self.train_x)
        self.inp_test = self.arch[0].forwardPropagate(self.test_x)
        fp = self.inp_train
        for layer in self.arch[1:]:
            fp = layer.forwardPropagate(fp)

    def train(self):

        for _ in trange(epochs):
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

    def layerFactory(self, classStr):
        if (v := classStr.split())[0] == "FullyConnected":
            return FullyConnectedLayer(int(v[1]), int(v[2]), learning_rate)
        elif classStr == "Sigmoid":
            return SigmoidLayer(None)
        elif classStr == "ReLu":
            return ReLuLayer(None)
        elif classStr == "Softmax":
            return SoftmaxLayer(None)
        elif classStr == "Tanh":
            return TanhLayer(None)
        elif classStr == "CrossEntropy":
            return CrossEntropy(self.train_y)
        elif classStr == "LogLoss":
            return LogLoss(self.train_y)
        elif classStr == "LeastSquares":
            return LeastSquares(self.train_y)
        elif classStr == "Input":
            return InputLayer(self.train_x)
        else:
            raise AttributeError

    def calculate_Accuracies(self):
        total_train = np.sum(np.equal(tyh := one_hot_decoder(self.train_y_hat), one_hot_decoder(self.train_y)))
        total_test = np.sum(np.equal(vyh := one_hot_decoder(self.test_y_hat), one_hot_decoder(self.test_y)))
        print(f'Train Accuracies: {total_train * 100 / tyh.shape[0]}%')
        print(f'Validation Accuracies: {total_test * 100 / vyh.shape[0]}%')
        # print(f'\nTraining Accuracy: {accuracy_score(one_hot_decoder(self.train_y), self.train_y_hat):.2f}')

