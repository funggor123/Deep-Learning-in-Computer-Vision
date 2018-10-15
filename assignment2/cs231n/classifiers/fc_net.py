from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = { }
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # Mean and SD
        mean = 0
        sd = weight_scale
        # First Layer Parameter Initlization 
        self.params['W1'] = np.random.normal(mean,sd,(input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        # Second Layer Parameter Initlization 
        self.params['W2'] = np.random.normal(mean,sd,(hidden_dim,num_classes))
        self.params['b2'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        cache = {}
        # First Layer Forward Pass
        a, cache['y1'] = affine_forward(X,self.params['W1'],self.params['b1'])
        h, cache['h1'] = relu_forward(a)
        # Second Layer Forward Pass 
        scores, cache['y2'] = affine_forward(h,self.params['W2'],self.params['b2']) 
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Data Loss
        data_loss ,dout = softmax_loss(scores,y)     
        # Regulatization Loss
        reg_loss = np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2'])
        # Total Loss
        loss = data_loss + reg_loss*self.reg*0.5
        
        # Backward Pass
        dout, grads['W2'], grads['b2'] = affine_backward(dout,cache['y2'])
        dout = relu_backward(dout, cache['h1'])
        dout, grads['W1'], grads['b1'] = affine_backward(dout,cache['y1']) 
        
        # Regularization Gradients
        grads['W2'] += self.params['W2']*2*self.reg*0.5
        grads['W1'] += self.params['W1']*2*self.reg*0.5
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        mean = 0
        sd = weight_scale  
        # Init by normal disturbution with mean = 0 and sd = weight_scale for W
        # Init by zero for b
        # First Layer Parameter Initialization
        # W1 = input_dim * hidden_dims[0]
        self.params['W1'] = np.random.normal(mean,sd,(input_dim,hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        # Middle Layer Parameter Initialization
        # WX = hidden_dims[ind-1] * hidden_dims[ind] 
        for layer_ind in range(1,len(hidden_dims)):
            self.params['W'+ str((layer_ind+1))] =  np.random.normal(mean,sd,(hidden_dims[layer_ind-1],hidden_dims[layer_ind]))
            self.params['b'+ str((layer_ind+1))] = np.zeros(hidden_dims[layer_ind])
        # Last Layer Parameter Initializaion
        # WN = hidden_dims[hidden_len-1] * num_classes
        self.params[ 'W'+ str(self.num_layers) ] = np.random.normal(mean,sd,(hidden_dims[len(hidden_dims)-1],num_classes))
        self.params[ 'b' + str(self.num_layers)] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train' } for i in range(self.num_layers - 1)]
            # beta and gamma initlizations 
            for layer_ind in range(0,len(hidden_dims)):
                self.params['gamma'+ str(layer_ind+1)] = np.ones(hidden_dims[layer_ind])
                self.params['beta'+ str(layer_ind+1)] = np.zeros(hidden_dims[layer_ind])
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        # Forward Pass
        # First Layer Forward Passing
        cache = {}
        t, cache['y1'] = affine_forward(X,self.params['W1'],self.params['b1'])
        if self.use_batchnorm:
            t, cache['bn1'] = batchnorm_forward(t,self.params['gamma1'],self.params['beta1'],self.bn_params[0])
        h, cache['h1'] = relu_forward(t)
        if self.use_dropout:
            h , cache['d1'] = dropout_forward(h,self.dropout_param)
        # Middle Layer Forward Passing
        for layer_ind in range(2,self.num_layers):
            t, cache[ 'y'+ str(layer_ind) ] = affine_forward(h,self.params['W'+ str(layer_ind)],self.params['b'+ str(layer_ind)])
            if self.use_batchnorm:
                t, cache['bn'+ str(layer_ind)] = batchnorm_forward(t,self.params['gamma'+ str(layer_ind)],self.params['beta'+ str(layer_ind)],self.bn_params[layer_ind-1])
            h, cache[ 'h'+ str(layer_ind) ] = relu_forward(t)
            if self.use_dropout:
                h , cache['d'+ str(layer_ind)] = dropout_forward(h,self.dropout_param)
        # Last Layer Forward Passing
        scores, cache[ 'y'+ str(self.num_layers) ] = affine_forward(h,self.params['W'+ str(self.num_layers)],self.params['b'+ str(self.num_layers)])              
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Loss Caculation 
        # Data loss
        data_loss ,dout = softmax_loss(scores,y) 
        # Regularization Loss
        reg_loss = 0.0
        for layer_ind in range(1,self.num_layers+1):
            reg_loss += np.sum(self.params['W'+ str(layer_ind)]*self.params['W'+ str(layer_ind)]) 
            reg_loss += np.sum(self.params['b'+ str(layer_ind)]*self.params['b'+ str(layer_ind)])
        # Total Loss
        loss = data_loss + reg_loss*self.reg*0.5
        
        # Backward Pass
        # last Layer Backward Pass
        dout, grads['W'+ str(self.num_layers)], grads['b'+ str(self.num_layers)] = affine_backward(dout,cache[ 'y'+ str(self.num_layers)])
        # Middle to first Layer Backward Pass
        for layer_ind in range(self.num_layers-1,0,-1):
            if self.use_dropout:
                dout = dropout_backward(dout, cache[ 'd'+ str(layer_ind) ])
            dout = relu_backward(dout, cache[ 'h'+ str(layer_ind) ])
            if self.use_batchnorm:
                dout, grads['gamma'+ str(layer_ind)], grads['beta'+ str(layer_ind)] = batchnorm_backward(dout,cache['bn'+str(layer_ind)])
            dout, grads['W'+ str(layer_ind)], grads['b'+ str(layer_ind)] = affine_backward(dout,cache[ 'y'+ str(layer_ind) ])           
        # Regularization Gradient
        for layer_ind in range(1,self.num_layers+1):
            grads['W'+ str(layer_ind)] += self.params['W'+ str(layer_ind)]*2*self.reg*0.5
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
