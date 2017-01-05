import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from importlib import reload
from scipy.optimize import fmin_cg as fmincg

def estimateGaussian(X):
    #ESTIMATEGAUSSIAN This function estimates the parameters of a 
    #Gaussian distribution using the data in X
    m, n = X.shape
    mu = X.mean(axis=0)
    sigma2 = np.mean((X-mu)**2,axis=0)
    return mu.reshape([1,-1]), sigma2.reshape([1,-1])

def multivariateGaussian(X, mu, Sigma2):
    #MULTIVARIATEGAUSSIAN Computes the probability density function of the
    #multivariate gaussian distribution.
    k = mu.size

    if 1 in Sigma2.shape:
        Sigma2 = np.diag(Sigma2[0])

    X =  X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5) * \
        np.exp(-0.5 * np.sum(X.dot(np.linalg.inv(Sigma2))*(X),axis=1))
    return p

def visualizeFit(X, mu, sigma2):
    #VISUALIZEFIT Visualize the dataset and its estimated distribution.

    X1,X2 = np.meshgrid(np.arange(0,35,.5),np.arange(0,35,.5)) 
    Z = multivariateGaussian(np.vstack([X1.ravel(),X2.ravel()]).T,mu,sigma2)
    Z = np.reshape(Z,X1.shape)

    plt.plot(X[:, 0], X[:, 1],'bx')

    # Do not plot if there are infinities
    if (np.sum(Z==np.inf) == 0):
        plt.contour(X1, X2, Z, levels=np.logspace(-20,0,8),c='k')
    plt.axis('square')
    plt.axis([0,30,0,30])
    
def selectThreshold(yval, pval):
    #SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    #outliers
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval),np.max(pval),stepsize):    
        predict = np.where(pval < epsilon,1,0)
        tp = np.sum((yval==predict) & (yval == 1))
        fp = np.sum((yval != predict) & (yval == 0))
        fn = np.sum((yval != predict) & (yval == 1))
    
        prec = tp/(tp+fp)
        rec  = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
            
    return bestEpsilon, bestF1

def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    #cell array of the words

    # Read the fixed movieulary list

    # Store all movies in cell array movie{}
    n = 1682  # Total number of movies 

    movieList = dict()
    with open('movie_ids.txt',encoding='cp1252') as fid:
        for i in range(n):
            # Read line
            line = fid.readline()
            movieName =' '.join(line.split(' ')[1:-1])
            # Actual Word
            movieList[i] = movieName.strip()
            
    return movieList

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lam):
    #COFICOSTFUNC Collaborative filtering cost function

    # Unfold the U and W matrices from params
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))
            
    J=0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    M = (X.dot(Theta.T) - Y)*R
    J = 0.5*np.sum(M**2) + 0.5*lam*np.sum(Theta**2) + 0.5*lam*np.sum(X**2) 

    X_grad = M.dot(Theta) + lam*X
    Theta_grad =  M.T.dot(X) + lam*Theta

    grad = np.hstack([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad

def normalizeRatings(Y, R):
    #NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
    #movie (every row)
    m, n = Y.shape
    Ymean = np.zeros([m, 1])
    Ynorm = np.zeros_like(Y)
    for i in range(m):
        idx = np.argwhere(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        
    return Ynorm, Ymean