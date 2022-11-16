# module containing various functions for use with an artificial neural
# network implementation
import numpy as np
rng = np.random.default_rng()


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


def softmax(V):
    # predict against NaNs by subtracting by max value
    return (np.exp(V - np.max(V)) / (np.exp(V - np.max(V)).sum()))


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


def binaryCrossEntropy(targ, pred):
    # add small value to prevent divide by 0 errors
    return -1. * np.sum(
        (np.array([1 if idx == targ else 0 for idx in range(2)]))
        * np.log(np.array([1 - pred[0], pred[0]]) + 1e-15)
    )


def multiCrossEntropy(targ, pred):
    # add small value to prevent divide by 0 errors
    return -1 * np.sum(
        (np.array([1 if idx == targ else 0 for idx in range(pred.shape[0])]))
        * np.log(pred + 1e-15)
    )


# other
########
# creates dataset of specified size with step between mean of each cloud in
# every dimension
def createDataset_clouds(inputCountPerClass,
                         variableCount,
                         classCount,
                         step=1,
                         spread=1,
                         ):
    data = np.zeros((inputCountPerClass * classCount, variableCount + 1))
    for aClass in range(classCount):
        for variable in range(variableCount):
            data[inputCountPerClass * aClass:
                 inputCountPerClass * (aClass + 1),
                 variable] = np.random.normal(
                     (aClass * step),
                     spread,
                     inputCountPerClass
                     )
        data[inputCountPerClass * aClass:
             inputCountPerClass * (aClass + 1), -1] = float(aClass)
    rng.shuffle(data)
    return data


def createDecisionBoundaryMesh(
    type,
    model,
    data,
    xSteps=60,
    ySteps=60,
):
    xTrueRange = data[:, 0].max() - data[:, 0].min()
    yTrueRange = data[:, 1].max() - data[:, 1].min()
    extra = 0.1
    xPlotRange = xTrueRange * (1. + extra)
    yPlotRange = yTrueRange * (1. + extra)
    x = np.arange(
        data[:, 0].min() - 0.5 * extra * xPlotRange,
        data[:, 0].max() + 0.4 * extra * xPlotRange,
        xPlotRange / xSteps,
        )
    y = np.arange(
        data[:, 1].min() - 0.5 * extra * yPlotRange,
        data[:, 1].max() + 0.4 * extra * yPlotRange,
        yPlotRange / ySteps,
        )
    # create meshgrid arrays
    X, Y = np.meshgrid(x, y)
    # create empty array to contain all meshgrid points
    flattenedCoords = np.zeros(((xSteps)*(ySteps), 2))
    flattenedCoords[:, 0] = X.flatten().copy()
    flattenedCoords[:, 1] = Y.flatten().copy()
    flattenedClassLabels = []
    if type == 'multiClass':
        flattenedConfidence = []
        for idx in range(flattenedCoords.shape[0]):
            predY = model.forwardPass(flattenedCoords[idx])
            flattenedClassLabels.append(np.argmax(predY))
            flattenedConfidence.append(np.max(predY))
        classLabels = np.array(flattenedClassLabels).reshape((xSteps, ySteps))
        confidence = np.array(flattenedConfidence).reshape((xSteps, ySteps))
        return X, Y, classLabels, confidence
    elif type == 'binaryClass':
        for idx in range(flattenedCoords.shape[0]):
            flattenedClassLabels.append(
                model.forwardPass(flattenedCoords[idx]))
        classPredict = np.array(flattenedClassLabels).reshape((xSteps, ySteps))
        return X, Y, classPredict
    else:
        print('Nahhh, only "multiClass" or "binaryClass" please')
