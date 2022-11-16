# module containing artificial neural network structure classes
import numpy as np
from functions import *
rng = np.random.default_rng()


class Node:
    """
    Class containing neuron functionality

    Attributes
    ----------
    Wij : numpy array
        node input weights
    delWij_n_1 : numpy array
        node input weight change from previous weight update
    actFunc : function
        computes node output from induced local field
    actFuncGrad : function
        computes gradient of node output for backpropagation
    self.Yi : numpy array
        vector input to node including constant value of 1. for bias
        multiplication at position 0
    self.vj : float
        induced local field, sum(inputs x weights)
    self.yj : float
        node output computed during forward pass

    """
    def __init__(self, numberInputs, actFunc):
        """
        Node initialisation, randomly assigns node weights

        Parameters
        ----------
        numberInputs : int
            number of inputs to node, excluding bias
        actFunc : string
            name of activation function to be used for node:
            'relu', 'sigmoid', 'linear'

        """
        # initialise node weights and bias
        self.Wij = np.random.normal(0.0, 1.0, numberInputs+1)
        self.delWij_n_1 = np.zeros(numberInputs + 1)
        # assign activation function and gradient from functions module
        self.actFunc = globals()[actFunc]
        self.actFuncGrad = globals()[actFunc + 'Grad']

    def computeNodeOutput(self, X):
        """
        Computes node output during forward pass

        Parameters
        ----------
        X : numpy array
            input to node during forward pass

        Returns
        -------
        yj : float
            node output

        """
        # insert constant value of 1. at position 0 for bias
        # store intermediate computed values for backpropagation
        self.Yi = np.insert(X, 0, 1.)
        self.vj = self.computeInducedLocalField()
        self.yj = self.actFunc(self.vj)
        return self.yj

    def computeInducedLocalField(self):
        """ Computes induced local field as sum(inputs x weights) """

        return np.inner(self.Yi, self.Wij)

    def computeOutputDelJ(self, ej):
        """
        Computes local gradient of loss function of output layer node

        Parameters
        ----------
        ej : float
            current node error

        """
        return self.actFuncGrad(self.yj) * ej

    def computeHiddenDelJ(self, DEL_K, Wkj):
        """
        Computes local gradient of loss function of hidden layer node

        Parameters
        ----------
        DEL_K : numpy array
            local gradients of loss function of subsequent layer nodes
        Wkj : numpy array
            connection weights from current node to subsequent layer nodes

        """
        return self.actFuncGrad(self.yj) * np.inner(DEL_K, Wkj)

    def weightUpdate(self, ej, DEL_K, Wkj, lr, mu):
        """
        Performs backpropagation weight update for current node

        Parameters
        ----------
        ej : float or None
            current node error
        DEL_K : numpy array or None
            local gradients of loss function of subsequent layer nodes
        Wkj : numpy array or None
            connection weights from current node to subsequent layer nodes
        lr : float
            weight update step size (learning rate)
        mu : float
            weight update momentum term

        Returns
        -------
        del_j : float
            current node local gradient of loss function
        Wij_n_1[1:] : numpy array
            current node input connection weights prior to weight update,
            do not pass up bias term at position 0

        """
        # local gradient computation different for output and hidden layers
        if ej:
            del_j = self.computeOutputDelJ(ej)
        else:
            del_j = self.computeHiddenDelJ(DEL_K, Wkj)
        # compute momentum term based on previous weight update step
        momentum = mu * self.delWij_n_1
        # compute weight update step
        delWij = lr * del_j * self.Yi + momentum
        # delWij = -1. * lr * del_j * self.Yi + momentum
        # store weight update step for next iteration's momentum term
        self.delWij_n_1 = delWij.copy()
        # store weights prior to update for preceeding layer's weight update
        Wij_n_1 = self.Wij.copy()
        # perform weight update step
        self.Wij += delWij
        # perform weights regularisation
        # self.Wij = self.Wij / np.linalg.norm(self.Wij, 2)
        return del_j, Wij_n_1[1:]


class Layer:
    def __init__(self, numberInputs, numberNodes, activationFunc):
        self.actFunc = activationFunc
        if activationFunc == 'softmax':
            self.nodes = [Node(numberInputs, 'linear')
                          for _ in range(numberNodes)]
        else:
            self.nodes = [Node(numberInputs, activationFunc)
                          for _ in range(numberNodes)]
        self.W_matrix = np.zeros((numberNodes, numberInputs))

    def computeLayerOutput(self, layerInputArray):
        if self.actFunc == 'softmax':
            layerOutputArray = np.array(
                [node.computeNodeOutput(layerInputArray)
                 for node in self.nodes])
            smax = softmax(layerOutputArray)
            return(smax)
        else:
            layerOutputArray = np.array(
                [node.computeNodeOutput(layerInputArray)
                 for node in self.nodes])
            return layerOutputArray

    def outputWeightUpdate(self, E, lr, mu):
        del_jList = []
        for idx in range(len(self.nodes)):
            del_j, Wij_n_1 = self.nodes[idx].weightUpdate(
                E[idx], None, None, lr, mu,
            )
            del_jList.append(del_j)
            # weights matrix contains this layers weights pre-update
            # each row corresponds to recieving node
            # each col corresponds to origin node
            self.W_matrix[idx] = Wij_n_1.copy()
        self.DEL_J = np.array(del_jList)
        return self.DEL_J, self.W_matrix

    def hiddenWeightUpdate(self, DEL_K, WKJ, lr, mu):
        del_jList = []
        for idx in range(len(self.nodes)):
            del_j, Wij_n_1 = self.nodes[idx].weightUpdate(
                None, DEL_K, WKJ[:, idx], lr, mu,
            )
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
    def __init__(self, inputLayer, hiddenLayers, outputLayer,
                 learningRate, momentum, lossFunction):
        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer
        self.lr = learningRate
        self.mu = momentum
        self.lossFunc = lossFunction

    def forwardPass(self, X):
        X = self.inputLayer.computeLayerOutput(X)
        for layer in self.hiddenLayers:
            X = layer.computeLayerOutput(X)
        Y = self.outputLayer.computeLayerOutput(X)
        return Y

    def backwardPass(self, E):
        DEL_K, WKJ = self.outputLayer.outputWeightUpdate(E, self.lr, self.mu)
        for layer in reversed(self.hiddenLayers):
            DEL_K, WKJ = layer.hiddenWeightUpdate(DEL_K, WKJ,
                                                  self.lr, self.mu)
        DEL_K, WKJ = self.inputLayer.hiddenWeightUpdate(DEL_K, WKJ,
                                                        self.lr, self.mu)

    def train(self, data, epochs, verbose=True):
        trainingHistory = []
        for epoch in range(epochs):
            totalEpochLoss = 0.
            rng.shuffle(data)
            for idx in range(data.shape[0]):
                X = data[idx, :-1]
                predY = self.forwardPass(X)
                targY = data[idx, -1]
                loss = self.lossFunc(targY, predY)
                totalEpochLoss += loss
                # multiclass
                if predY.shape[0] > 1:
                    targY = (np.array(
                        [1 if idx == targY
                            else 0 for idx in range(predY.shape[0])]))
                    self.backwardPass((targY - predY) * loss)
                # binary
                else:
                    self.backwardPass(np.sign(targY - predY) * loss)
            meanEpochLoss = totalEpochLoss / data.shape[0]
            trainingHistory.append(meanEpochLoss)
            if meanEpochLoss < 0.1:
                print(f'Early stopping at epoch: {epoch}')
                break
            if verbose:
                if epoch % int(epochs//5) == 0:
                    print(f'Epoch: {epoch}')
                    predArr = self.test(data[:, :-1])
                    print(f'Classification accuracy: \
{((predArr == data[:, -1]).sum() * 100 / data.shape[0]):.1f}%')
        return trainingHistory

    def test(self, X):
        predList = []
        for idx in range(X.shape[0]):
            predY = self.forwardPass(X[idx])
            # multiclass
            if predY.shape[0] > 1:
                predictedClass = np.argmax(predY)
            # binary
            else:
                if predY[0] > 0.5:
                    predictedClass = 1.
                else:
                    predictedClass = 0.
            predList.append(predictedClass)
        predArr = np.array(predList)
        return predArr

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
