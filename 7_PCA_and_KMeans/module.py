import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from IPython import display
from time import sleep


def findClosestCentroids(X, centroids):
    #FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    #   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    # Set K
    K = centroids.shape[0]

    idx = np.zeros([X.shape[0], 1])

    for i in range(X.shape[0]):
        idx[i] = np.argmin(np.sum((X[i,:] - centroids)**2,1))
    return idx


def computeCentroids(X, idx, K):
    #COMPUTECENTROIDS returs the new centroids by computing the means of the 
    #data points assigned to each centroid.
    m, n = X.shape
    centroids = np.zeros([K, n])

    for k in range(K):
        centroids[k,:] = np.mean(X[idx[:,0]==k,:],axis=0)

    return centroids

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color

    # Plot the data
    for k in range(K):
        indx = idx == k
        rgba = plt.cm.hsv(20*k)
        plt.scatter(X[indx.flatten(),0], X[indx.flatten(),1], s=20,c=rgba)

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    #PLOTPROGRESSKMEANS is a helper function that displays the progress of 
    #k-Means as it is running. It is intended for use only with 2D data.

    # Plot the examples
    for k in range(K):
        indx = idx == k
        rgba = plt.cm.hsv(int(255*k/K))
        plt.scatter(X[indx.flatten(),0], X[indx.flatten(),1], s=25,c=rgba)

    # Plot the centroids as black x's
    plt.plot(centroids[:,0], centroids[:,1], 'x',\
         markeredgecolor='k', \
         markersize = 10, linewidth = 3)

    # Plot the history of the centroids with lines
    
    for j in range(centroids.shape[0]):
        x= [previous[j,0],centroids[j,0]]
        y= [previous[j,1],centroids[j,1]]
        plt.plot(x, y,c='k')

    plt.title('Iteration number {0:d}'.format(i))
    
def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    #RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    #is a single example

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros([m, 1])
    
    # Run K-Means
    for i in range(max_iters):

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
    
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            display.clear_output(wait=True)
            display.display(plt.gcf())
            sleep(1)

            #input('Press enter to continue')
    
    
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)
        
    plt.close()
    return centroids,idx

def kMeansInitCentroids(X, K):
    #KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    #used in K-Means on the dataset X

    randindx = np.random.permutation(X.shape[0])
    return X[randindx[:K],:]

def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X 
    mu = X.mean(axis=0)
    X_norm =  X - mu
    sigma = X_norm.std()
    X_norm = X_norm/ sigma
    return X_norm,mu,sigma


def  pca(X):
    #PCA Run principal component analysis on the dataset X
    # Useful values
    m, n = X.shape

    SIGMA = X.T.dot(X)/m
    U,S,V = np.linalg.svd(SIGMA)
    return U,S

def drawLine(p1, p2,**kwargs):
    #DRAWLINE Draws a line from point p1 to point p2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def projectData(X, U, K):
    #PROJECTDATA Computes the reduced data representation when projecting only 
    #on to the top k eigenvectors
    return X.dot(U[:,:K])

def recoverData(Z, U, K):
    #RECOVERDATA Recovers an approximation of the original data when using the 
    #projected data
    return Z.dot(U[:,:K].T)

def displayData(X, example_width=None):
    #DISPLAYDATA Display 2D data in a nice grid

    # Set example_width automatically if not passed in
    if example_width is None: 
        example_width = int(np.sqrt(X.shape[1]))

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    
    fig,ax = plt.subplots(nrows=display_rows,ncols=display_cols)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    #fig.tight_layout
    
    k=0
    for j in range(display_rows):
        for i in range(display_cols):
            ax_c = ax[j,i]
            ax_c.set_axis_off()
            X_c = X[k,:].reshape([example_height,example_width])
            ax_c.imshow(X_c.T,cmap=plt.cm.gray)
            k +=1              