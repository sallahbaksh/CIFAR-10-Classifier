import pandas as pd
import numpy as np
from tqdm import trange
from Layers import *
from ObjectiveFuncs import *
from matplotlib import pyplot as plt

def one_hot_encoder(data):
    return np.squeeze(np.eye(np.max(data) + 1)[data.reshape(-1)])

class GAN:
    def __init__(self, eta, digit, train, labels, batch_size, epochs) -> None:

        # storing input parameters and data
        self.eta = eta
        self.train = train.values
        self.batch_size = batch_size
        self.digit = digit
        self.epochs = epochs
        self.labels = labels

        # selecting just the data needed for digit
        temp =  pd.DataFrame(self.train[self.train[:, 0] == digit])
        self.ytr = temp.pop(0).values
        self.xtr = temp.values
        self.batches = self.xtr.shape[0] // self.batch_size

        # initializing common layer
        self.common_fcl = FullyConnectedLayer(
            self.xtr.shape[1], self.xtr.shape[1], self.eta)
        self.common_tanh = TanhLayer(self.common_fcl.forwardPropagate(self.xtr))

        # initializing discriminator
        self.disc_fcl = FullyConnectedLayer(self.xtr.shape[1], 1, self.eta)
        self.disc_sigmoid = SigmoidLayer(self.disc_fcl.forwardPropagate(
            self.common_tanh.forwardPropagate(self.common_fcl.forwardPropagate(self.xtr))))
        self.disc_log_loss = None

        # initializing classifier
        self.class_fcl = FullyConnectedLayer(self.xtr.shape[1], 10, self.eta)
        self.class_sm = SoftmaxLayer(self.class_fcl.forwardPropagate(
            self.common_tanh.forwardPropagate(self.common_fcl.forwardPropagate(self.xtr))))
        self.class_ce = None

        # initializing generator
        self.gen_fcl = FullyConnectedLayer(
            self.xtr.shape[1], self.xtr.shape[1], self.eta)
        self.gen_relu = ReLuLayer(None)
        self.gen_logistic_loss = LogisticLoss()

        # lists to track loss
        self.gen_loss = []
        self.disc_loss = []
    
    def gen_forward_propagate(self,x):
        fcl_data = self.gen_fcl.forwardPropagate(x)
        return self.gen_relu.forwardPropagate(fcl_data)

    def gen_backward_propagate(self,y_pred):
        loss_grad = self.gen_logistic_loss.gradient(y_pred)
        d_fcl_grad = self.disc_fcl.gradient()
        d_grad = self.disc_sigmoid.backwardPropagate(loss_grad)

        g_loss = self.gen_relu.backwardPropagate(d_grad@d_fcl_grad)
        self.gen_fcl.simpleBackwardPropagate(g_loss, self.eta)

    def disc_forward_propagate(self,x):
        common_fcl = self.common_fcl.forwardPropagate(x)
        common_obj = self.common_tanh.forwardPropagate(common_fcl)

        class_fcl = self.class_fcl.forwardPropagate(common_obj)
        return self.class_sm.forwardPropagate(class_fcl)

    def class_forward_propagate(self, x):
        common_fcl = self.common_fcl.forwardPropagate(x)
        common_obj = self.common_tanh.forwardPropagate(common_fcl)

        fcl_data = self.disc_fcl.forwardPropagate(common_obj)
        return self.disc_sigmoid.forwardPropagate(fcl_data)

    def disc_backward_propagate(self,y_pred):
        ll_grad = self.disc_log_loss.gradient(y_pred)
        sl_grad = self.disc_sigmoid.backwardPropagate(ll_grad)
        fcl_grad = self.disc_fcl.simpleBackwardPropagate(sl_grad, self.eta)
        tanh_grad = self.common_tanh.backwardPropagate(fcl_grad)
        self.common_fcl.backwardPropagate(tanh_grad, self.eta)

    def class_backward_propagate(self, y_pred):
        ce_grad = self.class_ce.gradient(y_pred)
        sm_grad = self.class_sm.backwardPropagate(ce_grad)
        fcl_grad = self.class_fcl.simpleBackwardPropagate(sm_grad, self.eta)
        tanh_grad = self.common_tanh.backwardPropagate(fcl_grad)
        self.common_fcl.backwardPropagate(tanh_grad, self.eta)

    def gen_input(self,x):
        return np.random.RandomState(0).normal(np.mean(x), np.std(x, ddof=1), size=(self.batch_size, x.shape[1]))

    def train_model(self):

        bs = self.batch_size

        for i in trange(self.epochs):
            # for i in range(self.batches):
            slicex = self.xtr[i*bs:(i+1)*bs]
            slicey = self.ytr[i*bs:(i+1)*bs].reshape(bs, 1)
            slicey_label = one_hot_encoder(self.labels[i*bs:(i+1)*bs].reshape(bs,1))

            slicey[slicey == self.digit] = 1

            self.disc_log_loss = LogLoss(np.vstack((slicey,np.zeros((bs,1)))))
            self.class_ce = CrossEntropy(slicey_label)
            gen_output = self.gen_forward_propagate(self.gen_input(self.xtr))

            y_pred = self.disc_forward_propagate(np.vstack((slicex, gen_output)))
            y_pred_class = self.class_forward_propagate(
                np.vstack((slicex, gen_output)))

            self.disc_loss.append(self.disc_log_loss.eval(y_pred))

            self.disc_backward_propagate(y_pred) 
            self.class_backward_propagate(y_pred_class)
            gen_output = self.disc_forward_propagate(gen_output)
            self.gen_backward_propagate(gen_output)
            self.gen_loss.append(self.gen_logistic_loss.eval(gen_output))


    def display_graph(self):
        plt.plot([j for j in range(len(self.disc_loss))],
                 self.disc_loss, label='Discriminator Loss')
        plt.plot([j for j in range(len(self.gen_loss))],
                 self.gen_loss, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()





