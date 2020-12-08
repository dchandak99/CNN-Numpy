import numpy as np
import nn
import sys

from util import *
from layers import *

# +
np.random.seed(42)

# +
#np.random.randn(10, 10).shape
# -
def check_fully_connected():
    XTrain = np.random.randn(10, 100)
    YTrain = np.random.randn(10, 10)

    nn1 = nn.NeuralNetwork(10, 1)
    nn1.addLayer(FullyConnectedLayer(100, 10, 'softmax'))

    delta = 1e-7
    size = nn1.layers[0].weights.shape
    num_grad = np.zeros(size)
    
    #print(size)
    
    for i in range(size[0]):
        for j in range(size[1]):
            activations = nn1.feedforward(XTrain)
            #print(len(activations))
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[0].weights[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad[i, j] = num_grad_ij
            nn1.layers[0].weights[i, j] -= delta

    saved = nn1.layers[0].weights[:, :].copy()
    activations = nn1.feedforward(XTrain)
    nn1.backpropagate(activations, YTrain)
    new = nn1.layers[0].weights[:, :]
    ana_grad = saved - new

    print(np.linalg.norm(num_grad - ana_grad))
    assert np.linalg.norm(num_grad - ana_grad) < 1e-5
    print("Gradient Test Passed for Fully Connected Layer!")


def check_all_layers():
    XTrain = np.random.randn(10, 3, 32, 32)
    YTrain = np.random.randn(10, 10)

    nn1 = nn.NeuralNetwork(10, 1)

    nn1.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 4, 1, 'relu'))
    nn1.addLayer(AvgPoolingLayer([4, 30, 30], [2, 2], 2))
    nn1.addLayer(ConvolutionLayer([4, 15, 15], [4, 4], 4, 1, 'relu'))
    nn1.addLayer(MaxPoolingLayer([4, 12, 12], [2, 2], 2)) #original
    #nn1.addLayer(AvgPoolingLayer([4, 12, 12], [2, 2], 2))
    nn1.addLayer(FlattenLayer())
    nn1.addLayer(FullyConnectedLayer(144, 10, 'softmax'))

    delta = 1e-7
    size = nn1.layers[0].weights.shape
    num_grad = np.zeros(size)

    for a in range(size[0]):
        for b in range(size[1]):
            for i in range(size[2]):
                for j in range(size[3]):
                    activations = nn1.feedforward(XTrain)
                    loss1 = nn1.computeLoss(YTrain, activations)
                    nn1.layers[0].weights[a, b, i, j] += delta
                    activations = nn1.feedforward(XTrain)
                    loss2 = nn1.computeLoss(YTrain, activations)
                    num_grad_ij = (loss2 - loss1) / delta
                    num_grad[a, b, i, j] = num_grad_ij
                    nn1.layers[0].weights[a, b, i, j] -= delta

    saved = nn1.layers[0].weights[:, :, :, :].copy()
    activations = nn1.feedforward(XTrain)
    nn1.backpropagate(activations, YTrain)
    new = nn1.layers[0].weights[:, :, :, :]
    ana_grad = saved - new

    print(np.linalg.norm(num_grad - ana_grad))
    assert np.linalg.norm(num_grad - ana_grad) < 1e-5
    print("Gradient Test Passed for All layers!")

#np.random.randn(10, 100).shape

check_fully_connected()


check_all_layers()


