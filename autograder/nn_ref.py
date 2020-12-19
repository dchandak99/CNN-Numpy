import numpy as np
import random
from util import oneHotEncodeY
import itertools

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
		return -(Y*np.log(predictions[-1]+1e-10) + (1-Y)*np.log(1-predictions[-1]+1e-10)).sum()
		# raise NotImplementedError

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
		activations = [X.copy()]
		for layer in self.layers:
			X = layer.forwardpass(X)
			activations.append(X)
		return activations
		# raise NotImplementedError

	def backpropagate(self, activations, Y):
		# Input
		# activations : The activations at each layer(starting from second layer(first hidden layer)) of the
		# neural network calulated in the feedforward pass
		# Y : True labels of the training data
		# This method adjusts the weights(self.layers's weights) and biases(self.layers's biases) as calculated from the
		# backpropagation algorithm
		# Hint: Start with derivative of cross entropy from the last layer
		delta = -(Y/(activations[-1]+1e-10) - (1-Y)/(1-activations[-1]+1e-10))
		for l in range(len(self.layers)-1,-1,-1):
			delta = self.layers[l].backwardpass(self.alpha,activations[l],delta)
		# raise NotImplementedError
