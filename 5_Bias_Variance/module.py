import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from importlib import reload
import scipy.optimize as op

def linearRegCostFunction(X, y, theta, lam):
    #LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    #regression with multiple variables
    m = len(y) # number of training examples
    theta =  theta.reshape([-1,1])
    h = X.dot(theta)
    J = np.sum((h-y)**2)/2/m + lam*np.sum(theta[1:]**2)/2/m

    grad = ((h-y).T.dot(X)).T/m
    grad[1:] = grad[1:] + lam/m*theta[1:]
    #grad = grad.reshape([2,1])
    return (J,grad.ravel())

def trainLinearReg(X, y, lam):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lam

    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFun = lambda t: linearRegCostFunction(X, y, t, lam)[0]
    grad    = lambda t: linearRegCostFunction(X, y, t, lam)[1]
    # Minimize using fmincg
    theta = op.fmin_cg(f=costFun,x0=initial_theta,fprime=grad,disp=0)

    return theta

def learningCurve(X, y, Xval, yval, lam):
    #LEARNINGCURVE Generates the train and cross validation set errors needed 
    #to plot a learning curve
    m = len(y)
    error_train = np.zeros([m, 1])
    error_val   = np.zeros([m, 1])
    for i in range(1,m):
        theta = trainLinearReg(X[:i,:], y[:i], lam)
        error_train[i] = linearRegCostFunction(X[:i,:], y[:i], theta, 0)[0]
        error_val[i]   = linearRegCostFunction(Xval, yval, theta, 0)[0]
    return error_train,error_val



