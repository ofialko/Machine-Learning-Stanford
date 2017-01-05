import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def plotData(X,y):
    plt.plot(X,y,'r+',markersize=10,label='Training data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s');
    
def computeCost(X, y, theta):
    #  J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #  parameter for linear regression to fit the data points in X and y
    m = y.shape[0] # number of training examples
    h = X.dot(theta)
    #J = np.linalg.norm(h-y)**2/2/m
    J = np.sum((h-y)**2)/2/m
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    #taking num_iters gradient steps with learning rate alpha

    m = y.shape[0] # number of training examples
    J_history = np.zeros([num_iters, 1])
    for itr in range(num_iters):
        h=X.dot(theta)
        theta = theta - alpha*((X.T).dot(h-y))/m
     
        J_history[itr] = computeCost(X, y, theta)
        
    return theta, J_history

def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X 
    #FEATURENORMALIZE(X) returns a normalized version of X where
    #the mean value of each feature is 0 and the standard deviation
    #is 1. This is often a good preprocessing step to do when
    # working with learning algorithms.
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    return (X-mu)/sigma, mu, sigma
