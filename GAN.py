import numpy as np
from tqdm import trange
from Layers import *
from ObjectiveFuncs import *
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from softmax import Softmax


def one_hot_encoder(data):
    return np.squeeze(np.eye(np.max(data) + 1)[data.reshape(-1)])


class GAN:
    def __init__(self, eta, trainX_data, trainY_data, testX_data, testY_data, batch_size, epochs):

        # storing input parameters and data
        self.eta = eta
        self.train = trainX_data[:10000]
        self.batch_size = batch_size
        self.epochs = epochs

        self.ytr = trainY_data[:10000]
        self.xtr = trainX_data
        self.batches = self.xtr.shape[0] // self.batch_size

        xtr_temp = trainX_data[:100]

        self.yte = testY_data
        self.xte = testX_data

        # initializing common layer
        self.common_fcl = FullyConnectedLayer(
            self.xtr.shape[1], self.xtr.shape[1], self.eta)
        self.common_tanh = TanhLayer(
            self.common_fcl.forwardPropagate(xtr_temp))

        common_data = self.common_tanh.forwardPropagate(
            self.common_fcl.forwardPropagate(xtr_temp))

        # initializing discriminator
        self.disc_fcl = FullyConnectedLayer(self.xtr.shape[1], 1, self.eta)
        self.disc_sigmoid = SigmoidLayer(self.disc_fcl.forwardPropagate(
            common_data))
        self.disc_log_loss = None

        # initializing classifier
        self.class_fcl = FullyConnectedLayer(self.xtr.shape[1], 10, self.eta)
        # self.class_sm = SoftmaxLayer(self.class_fcl.forwardPropagate(
        #     common_data))
        self.class_sm = Softmax()
        self.class_ce_tr = None
        self.class_ce_te = None

        # initializing generator
        self.gen_fcl = FullyConnectedLayer(
            self.xtr.shape[1], self.xtr.shape[1], self.eta)
        self.gen_relu = ReLuLayer(None)
        self.gen_logistic_loss = LogisticLoss()

        # lists to track loss
        self.gen_loss = []
        self.disc_loss = []
        self.class_loss_tr = []
        self.class_loss_te = []

    def gen_forward_propagate(self, x):
        fcl_data = self.gen_fcl.forwardPropagate(x)
        return self.gen_relu.forwardPropagate(fcl_data)

    def gen_backward_propagate(self, y_pred):
        loss_grad = self.gen_logistic_loss.gradient(y_pred)
        d_fcl_grad = self.disc_fcl.gradient()
        d_grad = self.disc_sigmoid.backwardPropagate(loss_grad)

        g_loss = self.gen_relu.backwardPropagate(d_grad@d_fcl_grad)
        self.gen_fcl.simpleBackwardPropagate(g_loss, self.eta)

    def class_forward_propagate(self, x):
        common_fcl = self.common_fcl.forwardPropagate(x)
        common_obj = self.common_tanh.forwardPropagate(common_fcl)

        class_fcl = self.class_fcl.forwardPropagate(common_obj)
        return self.class_sm.forward(class_fcl)

    def disc_forward_propagate(self, x):
        common_fcl = self.common_fcl.forwardPropagate(x)
        common_obj = self.common_tanh.forwardPropagate(common_fcl)

        fcl_data = self.disc_fcl.forwardPropagate(common_obj)
        return self.disc_sigmoid.forwardPropagate(fcl_data)

    def disc_backward_propagate(self, y_pred):
        ll_grad = self.disc_log_loss.gradient(y_pred)
        sl_grad = self.disc_sigmoid.backwardPropagate(ll_grad)
        fcl_grad = self.disc_fcl.simpleBackwardPropagate(sl_grad, self.eta)
        tanh_grad = self.common_tanh.backwardPropagate(fcl_grad)
        self.common_fcl.simpleBackwardPropagate(tanh_grad, self.eta)

    def class_backward_propagate(self, y_pred):
        ce_grad = self.class_ce_tr.gradient(y_pred)
        sm_grad = self.class_sm.backward(y_pred, self.slicey_ohe, ce_grad)
        # sm_grad = self.class_sm.backwardPropagate(ce_grad)
        fcl_grad = self.class_fcl.simpleBackwardPropagate(sm_grad, self.eta)
        tanh_grad = self.common_tanh.backwardPropagate(fcl_grad)
        self.common_fcl.simpleBackwardPropagate(tanh_grad, self.eta)

    def gen_input(self, x):
        return np.random.RandomState(0).normal(np.mean(x), np.std(x, ddof=1), size=(self.batch_size, x.shape[1]))

    def train_model(self):

        bs = self.batch_size

        for _ in range(self.epochs):
            for i in trange(self.batches):
                slicex = self.xtr[i*bs:(i+1)*bs]
                slicey = self.ytr[i*bs:(i+1)*bs].reshape(bs, 1)

                slicexte = self.xte
                sliceyte = self.yte

                self.slicey_ohe = one_hot_encoder(slicey)
                self.sliceyte_ohe = one_hot_encoder(sliceyte)

                self.disc_log_loss = LogLoss(
                    np.vstack((slicey, np.zeros((bs, 1)))))
                self.class_ce_tr = CrossEntropy(self.slicey_ohe)
                self.class_ce_te = CrossEntropy(self.sliceyte_ohe)
                gen_output = self.gen_forward_propagate(self.gen_input(self.xtr))

                y_pred = self.disc_forward_propagate(
                    np.vstack((slicex, gen_output)))
                self.disc_loss.append(self.disc_log_loss.eval(y_pred))

                self.disc_backward_propagate(y_pred)

                self.test_y_pred_class = self.class_forward_propagate(
                    slicexte)

                self.class_loss_te.append(
                    self.class_ce_te.eval(self.test_y_pred_class))

                self.train_y_pred_class = self.class_forward_propagate(
                    slicex)

                self.class_loss_tr.append(
                    self.class_ce_tr.eval(self.train_y_pred_class))

                self.class_backward_propagate(self.train_y_pred_class)
                gen_output = self.disc_forward_propagate(gen_output)
                self.gen_backward_propagate(gen_output)
                self.gen_loss.append(self.gen_logistic_loss.eval(gen_output))

    def display_graph(self):
        plt.plot([j for j in range(len(self.class_loss_tr))],
                 self.class_loss_tr, label='Training Loss')
        plt.plot([j for j in range(len(self.class_loss_te))],
                 self.class_loss_te, label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def train_accuracy(self):

        y_hats = np.argmax(self.train_y_pred_class, axis=1)

        y_vals = np.argmax(self.slicey_ohe, axis=1)

        return accuracy_score(y_vals, y_hats)*100

    def test_accuracy(self):

        y_hats = np.argmax(self.test_y_pred_class, axis=1)

        y_vals = np.argmax(self.sliceyte_ohe, axis=1)

        return accuracy_score(y_vals, y_hats)*100
