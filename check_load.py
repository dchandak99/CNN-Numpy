import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *

#np.random.seed(4) #not affecting results
#modelName = 'model.p'
modelName = 'model.p'
#modelName = 'models/model_2_seed.p'

nn1 = nn.NeuralNetwork(10, 0.1)
nn1.addLayer(ConvolutionLayer([3, 32, 32], [5, 5], 32, 3, 'relu'))
nn1.addLayer(AvgPoolingLayer([32, 10, 10], [4, 4], 2))
nn1.addLayer(FlattenLayer())
nn1.addLayer(FullyConnectedLayer(512, 10, 'softmax'))

#model = np.load(modelName, allow_pickle=True)

# model = []
# with open(modelName, 'rb') as f:
# 	while True:
# 		try:
# 			model.append(pickle.load(f))
# 		except EOFError:
# 			break

#with open(modelName, 'rb') as f:
#	model = pickle.load(f)

model = pickle.load(open(modelName, 'rb'))

k,i = 0,0
for l in nn1.layers:
	if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
		nn1.layers[i].weights = model[k]
		nn1.layers[i].biases = model[k+1]
		k+=2
	i+=1

print("Model Loaded... ")

XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
XTest = XTest[0:1000,:,:,:]
YTest = YTest[0:1000,:]

pred, acc = nn1.validate(XTest, YTest)
print('Test Accuracy ',acc)
