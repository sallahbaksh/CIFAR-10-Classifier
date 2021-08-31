from statistics import mean
from unicodedata import digit
import pandas as pd
import numpy as np
from torch import softmax
import imageio
from sympy import fps
from tqdm import trange
from fully_connected import fully_connected
from logistic_loss import logistic_loss
from matplotlib import pyplot as plt
from log_loss import log_loss
from relu_layer import relu_layer
from sigmoid_layer import sigmoid_layer


class GAN:
    def __init__(self, eta, digit, train, batch_size, epochs) -> None:

        # storing input parameters and data
        self.eta = eta
        self.train = train.values
        self.batch_size = batch_size
        self.digit = digit
        self.epochs = epochs

        # selecting just the data needed for digit
        temp =  pd.DataFrame(self.train[self.train[:, 0] == digit])
        self.ytr = temp.pop(0).values
        self.xtr = temp.values
        self.batches = self.xtr.shape[0] // self.batch_size

        # initializing common layer
        self.common_fcl = fully_connected(self.xtr.shape[1], self.xtr.shape[1])

        # initializing discriminator
        self.disc_fcl = fully_connected(self.xtr.shape[1], 1)
        self.disc_sigmoid = sigmoid_layer()
        self.disc_log_loss = None

        # initializing classifier
        self.class_fcl = fully_connected(self.xtr.shape[1],10)
        self.class_sm = softmax()
        

        # initializing generator
        self.gen_fcl = fully_connected(self.xtr.shape[1], self.xtr.shape[1])
        self.gen_relu = relu_layer()
        self.gen_logistic_loss = logistic_loss()

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
        self.gen_fcl.backwardPropagate(g_loss, self.eta)

    def disc_forward_propagate(self,x):
        fcl_data = self.disc_fcl.forwardPropagate(x)
        return self.disc_sigmoid.forwardPropagate(fcl_data)

    def disc_backward_propagate(self,y_pred):
        ll_grad = self.disc_log_loss.gradient(y_pred)
        sl_grad = self.disc_sigmoid.backwardPropagate(ll_grad)
        self.disc_fcl.backwardPropagate(sl_grad, self.eta)

    def gen_input(self,x):
        return np.random.RandomState(0).normal(np.mean(x), np.std(x, ddof=1), size=(self.batch_size, x.shape[1]))

    def train_model(self):

        bs = self.batch_size

        images = []

        for i in trange(self.epochs):
            # for i in range(self.batches):
            slicex = self.xtr[i*bs:(i+1)*bs]
            slicey = self.ytr[i*bs:(i+1)*bs].reshape(bs, 1)

            slicey[slicey == self.digit] = 1

            self.disc_log_loss = log_loss(np.vstack((slicey,np.zeros((bs,1)))))
            gen_output = self.gen_forward_propagate(self.gen_input(self.xtr))

            y_pred = self.disc_forward_propagate(np.vstack((slicex, gen_output)))

            image = np.reshape(gen_output[np.argmax(y_pred)-bs], (28,28))
            image = (image*255)/image.max()
            images.append(image.astype(np.uint8))

            self.disc_loss.append(self.disc_log_loss.eval(y_pred))

            self.disc_backward_propagate(y_pred) 
            gen_output = self.disc_forward_propagate(gen_output)
            self.gen_backward_propagate(gen_output)
            self.gen_loss.append(self.gen_logistic_loss.eval(gen_output))


        imageio.mimsave(f"../videos/GAN{self.digit}.mp4", images, fps=10, macro_block_size=1)


    def display_graph(self):
        plt.plot([j for j in range(len(self.disc_loss))],
                 self.disc_loss, label='Discriminator Loss')
        plt.plot([j for j in range(len(self.gen_loss))],
                 self.gen_loss, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()




