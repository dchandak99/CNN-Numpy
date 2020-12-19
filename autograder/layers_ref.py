'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):

        # TODO
        n = X.shape[0]  # batch size
        # INPUT activation matrix       :[n X self.in_nodes]
        # OUTPUT activation matrix      :[n X self.out_nodes]

        ###############################################
        # TASK 1 - YOUR CODE HERE
        self.data = X@self.weights + self.biases
        if self.activation == 'relu':
            self.data = relu_of_X(self.data) 
            return self.data
        elif self.activation == 'softmax':
            self.data = softmax_of_X(self.data) 
            return self.data
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        # TODO 
        n = activation_prev.shape[0] # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        if self.activation == 'relu':
            inp_delta = actual_gradient_relu_of_X(self.data, delta)
        elif self.activation == 'softmax':
            inp_delta = gradient_softmax_of_X(self.data, delta)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        w = self.weights.copy()
        gradient = inp_delta.T@activation_prev
        # for i in range(n):
            # if i%1000000 == 0:
                # print((lr*inp_delta[i].reshape((self.out_nodes,1))@activation_prev[i].reshape((1,self.in_nodes))).T)
        self.weights -= (lr*gradient.T)
        self.biases -= lr*inp_delta.sum(axis=0)
        # print((gradient))
        # print(self.weights-w)
        return inp_delta@(w.T)
        ###############################################
        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def getOriginalIndex(self,row,col):
        #Return centered index of original image given coordinates of next act
        return row*self.stride,row*self.stride+self.filter_row,col*self.stride,col*self.stride+self.filter_col
        # pass
    def forwardpass(self, X):
        # TODO

        n = X.shape[0]  # batch size
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        ###############################################
        # TASK 1 - YOUR CODE HERE
        self.data = np.zeros((n,self.out_depth,self.out_row,self.out_col))
        # for filter in range(self.out_depth):
        for row in range(self.out_row):
            for col in range(self.out_col):
                x1,x2,y1,y2 = self.getOriginalIndex(row,col)
                # print(np.tile(X[:,:,None,x1:x2,y1:y2],(1,1,self.out_depth,1,1)).shape,np.tile(self.weights[None,:,:,:,:],(n,1,1,1,1)).shape)
                self.data[:,:,row,col] = (np.tile(self.weights[None,:,:,:,:],(n,1,1,1,1))*np.tile(
                    X[:,None,:,x1:x2,y1:y2],(1,self.out_depth,1,1,1))).sum((-1,-2,-3))
                self.data[:,:,row,col] += self.biases
        if self.activation == 'relu':
            self.data = relu_of_X(self.data)
            return self.data
            # raise NotImplementedError
        elif self.activation == 'softmax':
            self.data = softmax_of_X(self.data)
            return self.data

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        n = activation_prev.shape[0] # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        output_delta = np.zeros((n,self.in_depth,self.in_row,self.in_col))
        if self.activation == 'relu':
            inp_delta = actual_gradient_relu_of_X(self.data, delta)
            # raise NotImplementedError
        elif self.activation == 'softmax':
            inp_delta = gradient_softmax_of_X(self.data, delta)
            raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        # assert self.data.shape == inp_delta.shape
        w = np.tile(self.weights[None,:,:,:,:],(n,1,1,1,1))
        gradwt = np.zeros(self.weights.shape)
        gradb = np.zeros(self.biases.shape)
        for row in range(self.out_row):
            for col in range(self.out_col):
                # print((np.tile(inp_delta[:,:,None,row,col][:,:,:,None,None],
                #   (1,1,self.in_depth,self.filter_row,self.filter_col))*w).sum(axis=1).shape)
                x1,x2,y1,y2 = self.getOriginalIndex(row,col)
                # print(output_delta[:,:,x1:x2,y1:y2].shape)
                delta_slice = np.tile(inp_delta[:,:,None,row,col][:,:,:,None,None],
                    (1,1,self.in_depth,self.filter_row,self.filter_col))
                output_delta[:,:,x1:x2,y1:y2] += (delta_slice*w).sum(axis=1)
                gradwt += (delta_slice*np.tile(
                    activation_prev[:,None,:,x1:x2,y1:y2],(1,self.out_depth,1,1,1))).sum(axis=0)
        gradb += inp_delta.sum(axis=(0,2,3))
        self.weights -= lr*gradwt
        self.biases -= lr*gradb
        return output_delta
        ###############################################
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for max_pooling layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

    def getOriginalIndex(self,row,col):
        #Return centered index of original image given coordinates of next act
        return row*self.stride,row*self.stride+self.filter_row,col*self.stride,col*self.stride+self.filter_col
        # pass

    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        n = X.shape[0]  # batch size
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        ###############################################
        # TASK 1 - YOUR CODE HERE
        output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
        for row in range(self.out_row):
            for col in range(self.out_col):
                x1,x2,y1,y2 = self.getOriginalIndex(row,col)
                output[:,:,row,col] = X[:,:,x1:x2,y1:y2].sum(axis=(-1,-2))/(self.filter_row*self.filter_col)
        return output
        # raise NotImplementedError
        ###############################################


    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        n = activation_prev.shape[0] # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        output_delta = np.zeros((n,self.in_depth,self.in_row,self.in_col))
        for row in range(self.out_row):
            for col in range(self.out_col):
                x1,x2,y1,y2 = self.getOriginalIndex(row,col)
                output_delta[:,:,x1:x2,y1:y2] += (np.tile(delta[:,:,row,col][:,:,None,None],(1,1,self.filter_row,self.filter_col))/(self.filter_row*self.filter_col))
        return output_delta
        # raise NotImplementedError
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for max_pooling layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

    def getOriginalIndex(self,row,col):
        #Return centered index of original image given coordinates of next act
        return row*self.stride,row*self.stride+self.filter_row,col*self.stride,col*self.stride+self.filter_col
        # pass

    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        n = X.shape[0]  # batch size
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]
        self.max_ind = np.zeros((n,self.out_depth,self.out_row,self.out_col))
        ###############################################
        # TASK 1 - YOUR CODE HERE
        output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
        for row in range(self.out_row):
            for col in range(self.out_col):
                x1,x2,y1,y2 = self.getOriginalIndex(row,col)
                output[:,:,row,col] = X[:,:,x1:x2,y1:y2].reshape((n,self.out_depth,self.filter_row*self.filter_col)).max(axis=-1)
                self.max_ind[:,:,row,col] = X[:,:,x1:x2,y1:y2].reshape((n,self.out_depth,self.filter_row*self.filter_col)).argmax(axis=-1)
        return output
        # raise NotImplementedError
        ###############################################


    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        n = activation_prev.shape[0] # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        output_delta = np.zeros((n,self.in_depth,self.in_row,self.in_col))
        for row in range(self.out_row):
            for col in range(self.out_col):
                x1,x2,y1,y2 = self.getOriginalIndex(row,col)
                temp_del = np.zeros((n,self.in_depth,self.filter_row*self.filter_col))
                for i in range(temp_del.shape[0]):
                    for j in range(temp_del.shape[1]):
                        temp_del[i,j,self.max_ind[i,j,row,col].astype('int')] = 1.
                temp_del = temp_del.reshape((n,self.in_depth,self.filter_row,self.filter_col))
                output_delta[:,:,x1:x2,y1:y2] += temp_del*np.tile(delta[:,:,row,col][:,:,None,None],(1,1,self.filter_row,self.filter_col))
        return output_delta
        # raise NotImplementedError
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # print(X.shape)
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    return np.maximum(0.0,X)
    # raise NotImplementedError
    
def gradient_relu_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    return delta*(X>0.0)
    #np.ones(X.shape)*(X>0.0)
    # raise NotImplementedError

def actual_gradient_relu_of_X(X,delta):
    return delta*(X>0.0)

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    return np.exp(X)/(np.expand_dims((np.exp(X).sum(axis=-1)),axis=-1))
    # raise NotImplementedError
    
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first
    batch,out = X.shape
    # jacobian = np.zeros((batch,out,out))
    Y = np.zeros(X.shape)
    for i in range(batch):
        Y[i] = delta[i,:]@(((np.diag(X[i])-(X[i,:].reshape(out,1)@X[i,:].reshape(1,out)))))
    return Y
    
