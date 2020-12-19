import numpy as np
import random
from util import oneHotEncodeY
import itertools

#a = np.array([[1,2],[3,4]])
#1.-a


class NeuralNetwork:

    def __init__(self, out_nodes, lr):
        # Method to initialize a Neural Network Object
        # Parameters
        # out_nodes - number of output nodes
        # alpha - learning rate
        # batchSize - Mini batch size
        # epochs - Number of epochs for training
        self.layers = []
        self.out_nodes = out_nodes
        self.alpha = lr

    def addLayer(self, layer):
        # Method to add layers to the Neural Network
        self.layers.append(layer)


    def computeLoss(self, Y, predictions):
        # Returns the crossentropy loss function given the prediction and the true labels Y
        # TODO 
        #loss = 0
        n = Y.shape[0]
        #e1 = -(np.sum(Y * np.log(predictions[-1])))
        
        loss = -1 * np.sum(Y * np.log(predictions[-1]))

        return loss/n
        #raise NotImplementedError

        # END TODO
    def computeAccuracy(self, Y, predLabels):
        # Returns the accuracy given the true labels Y and predicted labels predLabels
        correct = 0
        for i in range(len(Y)):
            if np.array_equal(Y[i], predLabels[i]):
                correct += 1
        accuracy = (float(correct) / len(Y)) * 100
        return accuracy

    def validate(self, validX, validY):
        # Input 
        # validX : Validation Input Data
        # validY : Validation Labels
        # Returns the validation accuracy evaluated over the current neural network model
        valActivations = self.feedforward(validX)
        pred = np.argmax(valActivations[-1], axis=1)
        validPred = oneHotEncodeY(pred, self.out_nodes)
        validAcc = self.computeAccuracy(validY, validPred)
        return pred, validAcc

    def feedforward(self, X):
        # Input
        # X : Current Batch of Input Data as an nparray
        # Output
        # Returns the activations at each layer(starting from the first layer(input layer)) to 
        # the output layer of the network as a list of np multi-dimensional arrays
        # Note: Activations at the first layer(input layer) is X itself     
        # TODO
        l = []
        l.append(X.copy())
        prev = l[-1]
        
        for layer in self.layers:
            l.append(layer.forwardpass(prev))
            prev = l[-1]
        
        
        return l
        # END TODO

    def backpropagate(self, activations, Y):
        # Input
        # activations : The activations at each layer(starting from second layer(first hidden layer)) of the
        # neural network calulated in the feedforward pass
        # Y : True labels of the training data
        # This method adjusts the weights(self.layers's weights) and biases(self.layers's biases) as calculated from the
        # backpropagation algorithm
        # Hint: Start with derivative of cross entropy from the last layer

        # TODO
        '''
        activation_last = activations[-1]
        n = Y.shape[0]
        
        #last_delta = activation_last - Y
        last_delta = np.zeros_like(Y)
        
        for i in range(Y.shape[0]):
            row_sum = np.sum(Y[i])
            for j in range(Y.shape[1]):
                last_delta[i][j] = -Y[i][j] + row_sum*activation_last[i][j]
            
        #last_delta = activation_last.copy()
        #y = Y.argmax(axis=1)
        #last_delta[range(n), y] = -1
        
        #ast_delta = last_delta/n
        #last delta is the gradient of error wrt last activation
        
        print("last del")
        print(last_delta)
        
        #layer.backwardpass(alpha, activation_prev, delta)
        #now go layer by layer
        delta = last_delta
        '''
        
        last_delta = -1 * Y / activations[-1]

        delta = last_delta

        i = len(activations)-1
        for layer in reversed(self.layers):
            delta = layer.backwardpass(self.alpha, activations[i-1], delta)
            i = i-1
            
        
        
        
        #raise NotImplementedError
        # END TODO



#a = np.array([1,2])
#b = np.array([3,4])
#a*b




