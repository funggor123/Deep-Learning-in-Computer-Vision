import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Numpy Array[row,column] , [[..],[..],[..],[..]]
  num_data = X.shape[0]
  
  for i in range(num_data):
      softmax_result = np.zeros(W.shape[1])
      softmax_result = np.exp(np.dot(X[i,:],W))/np.sum(np.exp(np.dot(X[i,:],W)))
      dW[:,y[i]] = dW[:,y[i]] + -((1-softmax_result[y[i]])*X[i,:])/num_data
      loss += -np.log(softmax_result[y[i]])
  pass
  dW = dW + ((2*reg)*W)
  loss = loss / num_data + np.sum(reg*W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  hidden_unit_outputs = np.dot(X,W)
  exp_outputs = np.exp(hidden_unit_outputs)
  softmax_outputs = exp_outputs/np.sum(exp_outputs,axis=1) 
  cross_entropy_loss = np.choose(y, softmax_outputs.T) 
  loss = np.sum(cross_entropy_loss) + reg*np.sum(W*W)
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

