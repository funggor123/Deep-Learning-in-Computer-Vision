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
  num_class = W.shape[1]
  for i in range(num_data):
      hidden_out = np.dot(X[i,:],W)
      hidden_out -=np.max(hidden_out)
      expo = np.exp(hidden_out)
      sum_expo = np.sum(np.exp(hidden_out))
      softmax = expo/sum_expo
      loss += -np.log(softmax[y[i]]) 
      # (D,C)
      for k in range(num_class):
          dW[:, k] += (softmax[k] - (k == y[i])) * X[i]
  pass
  loss /= num_data 
  loss += 0.5*reg*np.sum(W*W)
  
  dW /= num_data 
  dW += reg*W
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
  # (N*D) * (D*C) -> N*C
  num_data = X.shape[0]
  hidden_out = np.dot(X,W)
  hidden_out -= np.max(hidden_out, axis=1, keepdims=True) # max of every sample
  # Forward Pass
  # (N*C) / (N => N*1 => N*C) = N*C
  # np.choose choose one element from each column
  # (N*C) => (C*N) -> C
  # C -> Scalar
  exp = np.exp(hidden_out)
  exp_sum = np.sum(exp,axis=1, keepdims=True)
  softmax = exp/exp_sum

  entropy_loss = -np.log(softmax)
  
  loss = np.sum(np.choose(y,entropy_loss.T))
  loss /= num_data
  loss += 0.5*reg*np.sum(W*W)
  
  # Backward Pass
  one_hot_Y = np.zeros((y.size, y.max()+1))
  one_hot_Y[np.arange(y.size),y] = 1
  
  # (N*C) - (N => N*C) = (N*C) 
  # (N*D) => (D*N) * (N*C)  = (D*C)
  dW += np.dot(X.T,softmax-one_hot_Y)
  dW /= num_data
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

