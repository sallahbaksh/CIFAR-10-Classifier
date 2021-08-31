#!/bin/python3
import numpy as np
import numpy.typing as npt

EPSILON = 10 ** -7


class CrossEntropy:
    def __init__(self, y: npt.ArrayLike):
        self.y: npt.ArrayLike = y
        self.forward_prop = None

    def eval(self, yhat: npt.ArrayLike):
        return np.average(-np.sum((np.multiply(self.y, np.log(yhat + EPSILON))), axis=1))

    def gradient(self, yhat: npt.ArrayLike):
        return -np.divide(self.y, yhat + EPSILON)

    def forwardPropagate(self, input_layer: npt.ArrayLike):
        self.forward_prop: npt.ArrayLike = input_layer
        return self.forward_prop

    def __repr__(self):
        return self.__class__.__name__


class LogLoss:
    """The Log Loss Objective Class requires that the input to the eval function are values between 0 and 1.
    """

    def __init__(self, y: npt.ArrayLike):
        self.y: npt.ArrayLike = y
        self.forward_prop = None

    def eval(self, yhat: npt.ArrayLike):
        return np.average(-1 * ((self.y * np.log(yhat + EPSILON)) + ((1 - self.y) * np.log(1 - yhat + EPSILON))))

    def gradient(self, yhat: npt.ArrayLike):
        return -1 * (self.y - yhat) / (yhat * (1 - yhat) + EPSILON)

    def forwardPropagate(self, input_layer: npt.ArrayLike):
        self.forward_prop: npt.ArrayLike = input_layer
        return self.forward_prop

    def __repr__(self):
        return self.__class__.__name__


class LeastSquares:
    def __init__(self, y: npt.ArrayLike):
        self.y: npt.ArrayLike = y
        self.forward_prop = None

    def eval(self, yhat: npt.ArrayLike):
        return np.average((self.y - yhat) ** 2)

    def gradient(self, yhat: npt.ArrayLike):
        return -2 * (self.y - yhat)

    def forwardPropagate(self, input_layer: npt.ArrayLike):
        self.forward_prop: npt.ArrayLike = input_layer
        return self.forward_prop

    def __repr__(self):
        return self.__class__.__name__

class LogisticLoss:
    def __init__(self):
        self.X = None

    def forwardPropagate(self, Y):
        self.forward_prop = Y
        return self.forward_prop

    def eval(self, yhat):
        return np.average(-np.log(yhat + EPSILON))

    def gradient(self, yhat):
        return -np.divide(1, yhat + EPSILON)