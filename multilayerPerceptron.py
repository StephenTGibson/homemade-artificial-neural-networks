# module containing artificial neural network structure classes

import numpy as np
from functions import *

# 
class Node:

    def __init__(self, numberInputs, actFunc):

        # initialise weights
        # self.Wij = np.array([0.,0.,0.5])
        # self.Wij = np.random.uniform(-1.0, 1.3, numberInputs+1)
        # #np.random.rand(numberInputs+1) #rand
        self.Wij = np.random.normal(0.0, 1.0, numberInputs+1)
        self.actFunc = globals()[actFunc]
        self.actFuncGrad = globals()[actFunc + 'Grad']
        self.mu = 0.9 # momentum
        self.delWij_n_1 = np.zeros(numberInputs+1)

    def computeILF(self):
        return np.inner(self.Wij, self.Yi)

    def computeNodeOutput(self, X):
        self.Yi = np.insert(X, 0, 1.)
        self.vj = self.computeILF()
        self.yj = self.actFunc(self.vj)
        # print(self.yj)
        return self.yj

    def computeDelJ(self, ej, layerType, DEL_K, Wkj):
        # computes local gradient for current node (scalar)

        if layerType == 'hidden':
            del_j = self.actFuncGrad(self.yj) * np.inner(DEL_K, Wkj)
            # print(Wkj)
            # print(f'hidden layer del j: {del_j}')

        elif layerType == 'output':
            del_j = self.actFuncGrad(self.yj) * ej
            # print(f'output layer del j: {del_j}')

        return del_j

    def weightUpdate(self, e, lr, layerType, DEL_K, Wkj):
        # performs vector weight update for all weights of current node

        del_j = self.computeDelJ(e, layerType, DEL_K, Wkj) #weightUpdate(E[idx], lr, 'hidden', DEL_K, Wkj)

        # momentum
        momentum = 0.
        momentum = self.mu * self.delWij_n_1
        # compute delta weight vector
        delWij = -1. * lr * del_j * self.Yi + momentum
        # if np.any(delWij<0):
        #     print('here')
        # store delta weight term for next iteration momentum
        self.delWij_n_1 = delWij.copy()
        # store weights pre-update to pass back for previous layer weight update
        Wij_n_1 = self.Wij.copy()
        # perform weight update
        self.Wij += delWij
        # regularistation
        # self.Wij = self.Wij / np.linalg.norm(self.Wij, 2)

        # if np.any(self.Wij > Wij_n_1):
        #     print('new bigger than old')

        return del_j, Wij_n_1[1:] #Wij_n_1[1:] # don't want to pass up the bias

class Layer:
    def __init__(self, numberInputs, numberNodes, activationFunc):

        self.actFunc = activationFunc

        if activationFunc == 'softmax':
            self.nodes = [Node(numberInputs, 'linear') for n in range(numberNodes)]
        else:
            self.nodes = [Node(numberInputs, activationFunc) for n in range(numberNodes)]

        self.W_matrix = np.zeros((numberNodes, numberInputs))

    def computeLayerOutput(self, layerInputArray):
        if self.actFunc == 'softmax':
            layerOutputArray = np.array([node.computeNodeOutput(layerInputArray) for node in self.nodes])
            smax = self.softmax(layerOutputArray)
            return(smax)

        else:
            layerOutputArray = np.array([node.computeNodeOutput(layerInputArray) for node in self.nodes])
            return layerOutputArray

    def softmax(self, V):
        return (np.exp(V) / (np.exp(V).sum()))

    def outputWeightUpdate(self, E, lr):

        del_jList = []
        for idx in range(len(self.nodes)):
            del_j, Wij_n_1 = self.nodes[idx].weightUpdate(E[idx], lr, 'output', None, None)
            del_jList.append(del_j)
            # weights matrix contains this layers weights pre-update
            # each row corresponds to recieving node
            # each col corresponds to origin node
            self.W_matrix[idx] = Wij_n_1.copy()

        self.DEL_J = np.array(del_jList)


        return self.DEL_J, self.W_matrix

    def hiddenWeightUpdate(self, lr, DEL_K, WKJ):

        del_jList = []
        for idx in range(len(self.nodes)):
            del_j, Wij_n_1 = self.nodes[idx].weightUpdate(None, lr, 'hidden', DEL_K, WKJ[:, idx])
            del_jList.append(del_j)
            self.W_matrix[idx] = Wij_n_1.copy()
        self.DEL_J = np.array(del_jList)

        return self.DEL_J, self.W_matrix

    def __str__(self):

        str = ''

        for i, node in enumerate(self.nodes):
            str += f'Node {i} weights: {node.Wij}\n'

        return str

class MultilayerPerceptron:
    def __init__(self, type, inputLayer, hiddenLayers, outputLayer, learningRate, lossFunction):

        self.type = type

        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer

        self.lr = learningRate
        self.lossFunc = lossFunction

    def forwardPass(self, X):
        X = self.inputLayer.computeLayerOutput(X)
        for layer in self.hiddenLayers:
            X = layer.computeLayerOutput(X)
        Y = self.outputLayer.computeLayerOutput(X)
        return Y

    def backwardPass(self, E):

        DEL_K, WKJ = self.outputLayer.outputWeightUpdate(E, self.lr)
        for layer in reversed(self.hiddenLayers):
            DEL_K, WKJ = layer.hiddenWeightUpdate(self.lr, DEL_K, WKJ)
        DEL_K, WKJ = self.inputLayer.hiddenWeightUpdate(self.lr, DEL_K, WKJ)

    def train(self, data, epochs):

        trainingHistory = []

        for epoch in range(epochs):
            totalEpochLoss=0.

            for idx in range(data.shape[0]):

                X = data[idx, :-1]
                tru = data[idx, -1]
                Y = self.forwardPass(X)
                Y = np.array([1-Y[0], Y[0]])

                # create desired vector (0s and 1 at index of correct class)
                D = np.zeros(2)
                D[int(data[idx, -1])] = 1.

                # if epoch % int(epochs/5) == 0:
            #         print(f'Epoch: {epoch}')
            #         self.test(data)
                loss = self.lossFunc(D, Y)
                totalEpochLoss += loss

                self.backwardPass(np.array([np.sign(Y[1]-tru)*loss]))

            meanEpochLoss = totalEpochLoss/data.shape[0]
            trainingHistory.append(meanEpochLoss)
            if meanEpochLoss < 0.1:
                    print(f'Early stopping at epoch: {epoch}')
                    break
            # if epochs > 5:
            #     if epoch % int(epochs/5) == 0:
            #         print(f'Epoch: {epoch}')
            #         self.test(data)
        return trainingHistory

    def test(self, data):
        predList = []
        for idx in range(data.shape[0]):
            Y = self.forwardPass(data[idx, :-1])
            if Y[0] > 0.5:
                prediction = 1.
            else:
                prediction = 0.

            predList.append(prediction)
        predictions = np.array(predList)

        print(f'Classification accuracy: {(predictions == data[:, -1]).sum()*100 / data.shape[0]}%') #:.1f

        return predictions

    def __str__(self):

        str = ''
        print('Input layer:')
        print(self.inputLayer)
        for i, hiddenLayer in enumerate(self.hiddenLayers):
            print(f'Hidden layer {i+1}:')
            print(hiddenLayer)
        print('Output layer:')
        print(self.outputLayer)
        return str

