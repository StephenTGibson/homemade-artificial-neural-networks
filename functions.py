# module containing various functions for use with an artificial neural
# network implementation

import numpy as np

# activation functions
######################
# input: induced local field (sum of weights x inputs)
def linear(v):
    return v

def relu(v):
    if v < 0.:
        return 0.
    else:
        return v

def sigmoid(v):
        return 1. / (1. + np.exp(-1. * v))

# activation function gradients
###############################
# input: previous output of node
def linearGrad(y):
    return 1.

def reluGrad(y):
    if y < 0.:
        return 0.
    else:
        return 1.

def sigmoidGrad(y):
        return y * (1. - y)

# loss functions
################
# input: arrays of true and predicted values
def mae(true, predicted):
    return abs(true - predicted)

def crossEntropy(true, predicted):
    # add small value to prevent divide by 0 errors
    return -1.*np.sum((true) * np.log(predicted+1e-15))
