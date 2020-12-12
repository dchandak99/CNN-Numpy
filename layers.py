'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

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
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        #self.data = X@self.weights + self.biases
        
        
        if self.activation == 'relu':
            self.data = relu_of_X(X@self.weights + self.biases)
            return self.data
            
        elif self.activation == 'softmax':
            self.data = softmax_of_X(X@self.weights + self.biases)
            return self.data
            
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        # END TODO    
        
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        '''
        if self.activation == 'relu':
            #a = gradient_relu_of_X(activation_prev, delta)
            a = gradient_relu_of_X(self.data, delta)
        elif self.activation == 'softmax':
            #a = gradient_softmax_of_X(activation_prev, delta)
            a = gradient_softmax_of_X(self.data, delta)
        
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        print("I am A")
        print(a)
        
        m = activation_prev.shape[0]
  
        #X = activation_prev
        #b = np.multiply(activation_prev, a)
        b = activation_prev.T@a
        # ais dg/d(wtx+b) , we need dg/dw which is dg/d(wtx+b)* d(wx+b)/dw
        print("I am b")
        print(b)
        
        self.weights = self.weights - lr*b
        
        _db = np.sum(a, axis=0, keepdims=True)
        #_db = np.sum(a, axis=1)/m
        
        self.biases = self.biases-lr*_db
        
        self.data = activation_prev@self.weights + self.biases
        
        
        return a
        
        '''
        inp_delta = None

        if self.activation == 'relu':
            inp_delta = gradient_relu_of_X(self.data, delta)

        elif self.activation == 'softmax':
            inp_delta = gradient_softmax_of_X(self.data, delta)

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        _dw = np.mean(activation_prev[:, :, np.newaxis] @ inp_delta[:, np.newaxis, :], axis=0)
        _db = np.mean(inp_delta, axis=0).reshape([1, -1])

        del_error_prev = inp_delta @ self.weights.T
       
        self.biases = self.biases-lr*_db


        self.weights = self.weights - lr*_dw

        return del_error_prev
        
        # END TODO

#import numpy as np
#xx = np.array([[1,2],[3,4]])
#np.sum(xx, axis = 1, keepdims=True)


class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
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
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]
        
        # TODO
        n = X.shape[0]
        outpu = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col))
        
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col
                
                X_slice = X[:, :, i_start:i_end, j_start:j_end]
                #https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
                '''
                outpu[:, :, i, j] = np.sum(
                    X[:, :, i_start:i_end, j_start:j_end, np.newaxis] *
                    self.weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )
                '''
                flatt = X_slice.reshape([X_slice.shape[0], X_slice.shape[1], -1])
                outpu[:, :, i, j] = np.sum(np.einsum('nid,oid->nod', flatt,
                                                     self.weights.reshape(list(self.weights.shape[:2]) + [-1])), axis=-1)
                
        #self.data = outpu // check above is newaxis is needed or not
        #https://www.google.com/search?channel=fs&client=ubuntu&q=use+of+np+newaxis
        
        #self.data = outpu + self.biases
        outpu = outpu + self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        
        if self.activation == 'relu':
            self.data = relu_of_X(outpu)
            
            return self.data
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        #return self.data
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
        # TODO
        
        n = activation_prev.shape[0]   #batch size
        
        inp_delta = None
        ###############################################
        if self.activation == 'relu':
            inp_delta = gradient_relu_of_X(self.data.reshape([n, -1]), delta.reshape([n, -1])).reshape(self.data.shape)
            #inp_delta = gradient_relu_of_X(self.data, delta)
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
                
        #outpu = np.zeros_like(activation_prev)
        new_delta = np.zeros((n, self.in_depth, self.in_row, self.in_col))
        
        #_db = delta.sum(axis = (0, 2, 3)) / n
        #_dw = np.zeros_like(self.weights)
        
        
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col
                
                '''
                outpu[:, :, i_start:i_end, j_start:j_end] += np.sum(
                    self.weights[np.newaxis, :, :, :, :] *
                    delta[:, :, i:i+1, j:j+1, np.newaxis],
                    axis=4
                )
                
                _dw += np.sum(
                    activation_prev[:, :, i_start:i_end, j_start:j_end, np.newaxis] *
                    delta[:, :, i:i+1, j:j+1, np.newaxis],
                    axis=0
                )
                '''
                conv_back = np.einsum('no,oirc->nirc', inp_delta[:, :, i, j], self.weights)
                new_delta[:, :, i_start:i_end, j_start:j_end] = conv_back + new_delta[:, :, i_start:i_end, j_start:j_end]
                
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col

                _dw = np.mean(np.einsum('no,nirc->noirc', inp_delta[:, :, i, j], activation_prev[:, :, i_start:i_end, j_start:j_end]), axis=0)
                self.weights = self.weights - lr*_dw

                
        #new_delta = outpu[:, :, 0:h_in, 0:w_in]
        
        _db = np.mean(np.sum(inp_delta.reshape([n, self.out_depth, -1]), axis=-1), axis=0)
        self.biases = self.biases - lr*_db
        #self.weights = self.weights - lr*_dw
        
        #outpu = new delta
        '''
        inp_delta = None
        ###############################################
        if self.activation == 'relu':
            inp_delta = gradient_relu_of_X(self.data, delta)
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################
        '''
        return new_delta
        
        # END TODO


class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        outpu = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col))
        avg_ = np.ones([self.filter_row, self.filter_col]) / (self.filter_row * self.filter_col)
        
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col
                
                X_slice = X[:, :, i_start:i_end, j_start:j_end]
                
                #axis = -1 is the last dimension
                outpu[:, :, i, j] = np.sum(np.sum(X_slice * avg_[np.newaxis, np.newaxis, :, :], axis=-1), axis=-1)
        
        return outpu
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        n = activation_prev.shape[0]  # batch size
        new_delta = np.zeros((n, self.in_depth, self.in_row, self.in_col))
        avg_ = np.ones([self.filter_row, self.filter_col]) / (self.filter_row * self.filter_col)
        
        
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col
                
                #X_slice = activation_prev[:, :, i_start:i_end, j_start:j_end]
                conv_back = delta[:, :, i:i+1, j:j+1] * avg_[np.newaxis, np.newaxis, :, :]
                
                new_delta[:, :, i_start:i_end, j_start:j_end] = conv_back + new_delta[:, :, i_start:i_end, j_start:j_end] 
                
                #outpu[:, :, i_start:i_end, j_start:j_end] += \
                #    delta[:, :, i:i + 1, j:j + 1] * X_slice
        
        return new_delta
        
        # END TODO
        ###############################################

#import numpy as np
#np.ones([2,3])      #both work





class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        outpu = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col))
        self._cache = {}
        
        def _save_mask(x, cords):
            mask = np.zeros_like(x)
            n, c, h, w = x.shape
            x = x.reshape(n, c, h*w)
            idx = np.argmax(x, axis=2)

            n_idx, c_idx = np.indices((n, c))
            mask.reshape(n, c, h*w)[n_idx, c_idx, idx] = 1
            self._cache[cords] = mask

        
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col
                
                X_slice = X[:, :, i_start:i_end, j_start:j_end]
                _save_mask(x=X_slice, cords=(i, j))
                
                outpu[:, :, i, j] = np.max(X_slice, axis=(2, 3))
        
        return outpu
                
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        outpu = np.zeros_like(activation_prev)
        
        
        for i in range(self.out_row):
            for j in range(self.out_col):
                i_start = i*self.stride
                i_end = i_start + self.filter_row
                j_start = j*self.stride
                j_end = j_start + self.filter_col
                
                X_slice = activation_prev[:, :, i_start:i_end, j_start:j_end]
                
                outpu[:, :, i_start:i_end, j_start:j_end] = delta[:, :, i:i + 1, j:j + 1] * self._cache[(i, j)] + outpu[:, :, i_start:i_end, j_start:j_end]
        
        return outpu
        
        
        
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        n = X.shape
        return X.reshape((n[0], n[1] * n[2] * n[3] )) # flatten layer
        
        # print(X.shape)
        pass
    def backwardpass(self, lr, activation_prev, delta):
        
        return delta.reshape(activation_prev.shape)
        
        pass
        # END TODO

#a = np.array([[1,2], [3,4]])
#a.reshape(1,4)



'''
import numpy as np
a = np.array([[1,2],[3,4]])
n = a.shape
b = a.reshape((n[0] * n[1], ))
b
'''
#a = 0
#a += \
#    3
#a+1

# +
#b.reshape(a.shape)
# -





# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    
    return X * (X > 0)
    
    
    #raise NotImplementedError
    # END TODO 


#a = np.array([[1,-3,4],[2,-6,7]])
#a * (a > 0)





def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    a = 1. * (X > 0)
    return  np.multiply(a, delta)
    #return a
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    #softmax only cares about the relative differences in the elements of x\mathbf xx. 
    #That means we can protect softmax from overflow by subtracting the maximum element
    #raise NotImplementedError
    #exp = np.exp(x) # exp just calculates exp for all elements in the matrix
    #return exp / exp.sum(1)        # 1 is axis (1 is row wise)
    
    def softmax(x):
        #x = x - np.max(x)
        row_sum = np.sum(np.exp(x))
        return np.array([np.exp(x_i) / row_sum for x_i in x])#, dtype=np.float128)
    
    
    #orig_X = X.copy()
    row_maxes = np.max(X, axis=1)
    row_maxes = row_maxes[:, np.newaxis]  # for broadcasting
    X = X - row_maxes
    
    
    return np.array([softmax(row) for row in X])#, dtype=np.float128)
    
    
    
    # END TODO  


def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    def jacobian_softmax(s):      #Return the Jacobian matrix of softmax vector s
        return np.diag(s) - np.outer(s, s)
    #raise NotImplementedError
    #a = grad
    
    #s = softmax_of_X(X)
    s = X.copy()
    j = np.array([jacobian_softmax(row) for row in s])
    #print(j.shape)
    
    # a is ndarray of shape (batch_Size, batch_size, out_nodes, out_nodes)
    l = []
    #aa = []
    for i in range(0, X.shape[0]):
        #j(i, i) = jacobian of this example
        aa = np.matmul(delta[i], j[i])
        #print(aa)
        #break
        #l.append(aa)
        l.append(aa)
        
    a = np.array(l)
    
    #a = s*(1.-s)
    return a
    #return  a
    # END TODO


# +
#XTrain = np.random.randn(10, 10)

# +
#np.zeros((10, 10))
# -

'''
activation_last = softmax_of_X(XTrain)
Y = np.random.randn(10, 10)
n = Y.shape[0]

last_delta = activation_last - Y
        #last_delta = activation_last.copy()
        #y = Y.argmax(axis=1)
        #last_delta[range(n), y] = -1
        
last_delta = last_delta/n
        #last delta is the gradient of error wrt last activation
        
        #layer.backwardpass(alpha, activation_prev, delta)
        #now go layer by layer
delta = last_delta
'''

# +
#a=gradient_softmax_of_X(XTrain, delta)
# -

#m = XTrain.shape[0]
        
        #X = activation_prev
        #b = np.multiply(activation_prev, a)
#b = XTrain.T@a/m
        # ais dg/d(wtx+b) , we need dg/dw which is dg/d(wtx+b)* d(wx+b)/dw
        #self.weights = self.weights - lr*b

# +
#b
# -




