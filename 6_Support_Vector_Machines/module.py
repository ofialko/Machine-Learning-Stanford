import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from importlib import reload
import sys, re
import nltk
from nltk.stem.porter import PorterStemmer

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Find Indices of Positive and Negative Examples
    pos = y[:,0] == 1; neg = y[:,0] == 0
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+',linewidth =  1, markersize = 7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko',mfc = 'y', markersize = 7)



def linearKernel(x1, x2):
    #LINEARKERNEL returns a linear kernel between x1 and x2
   
    # Ensure that x1 and x2 are 1D vectors
    x1 = x1.ravel(); x2 = x2.ravel()
    return  x1.dot(x2)

def gaussianKernel(x1, x2, sigma):
    #RBFKERNEL returns a radial basis function kernel between x1 and x2

    # Ensure that x1 and x2 are 1D vectors
    #x1 = x1.ravel(); x2 = x2.ravel()
    return np.exp(-np.sum((x1-x2)**2)/2/sigma**2)

def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary 
    #learned by the SVM

    w = model['w']
    b = model['b']
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = - (w[0]*xp + b)/w[1]
    plt.plot(xp, yp, '-b')
    plotData(X, y);

def svmTrain(X, Y, C, kernelFunction,tol=None, max_passes=None,sigma=None):
    #SVMTRAIN Trains an SVM classifier using a simplified version 
    #of the SMO algorithm at http://cs229.stanford.edu/materials/smo.pdf
    
    #   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
    #   SVM classifier and returns trained model. X is the matrix of training 
    #   examples.  Each row is a training example, and the jth column holds the 
    #   jth feature.  Y is a column matrix containing 1 for positive examples 
    #   and 0 for negative examples.  C is the standard SVM regularization 
    #   parameter.  tol is a tolerance value used for determining equality of 
    #   floating point numbers. max_passes controls the number of iterations
    #   over the dataset (without changes to alpha) before the algorithm quits.

    if tol is None:
        tol = 1e-3
    
    if max_passes is None:
        max_passes = 5

    # Data parameters
    m,n = X.shape
    Y = Y.reshape([-1,1]) # ensure y is a column vector
    # Map 0 to -1
    Y[Y[:,0]==0,0] = -1

    # Variables
    alphas = np.zeros([m,1])
    E = np.zeros([m,1])
    b,passes,eta,L,H = 0,0,0,0,0

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    #  gracefully will _not_ do this)

    if kernelFunction == 'linearKernel':
        # Vectorized computation for the Linear Kernel
        # This is equivalent to computing the kernel on every pair of examples
        kernelFunction =  linearKernel
        K = X.dot(X.T)
    elif kernelFunction == 'gaussianKernel':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        kernelFunction = gaussianKernel
        X2 = np.sum(X**2, axis=1).reshape([-1,1])
        K = X2.T - 2*(X.dot(X.T))
        K = X2 + K
        K = gaussianKernel(1, 0,sigma)**K
    else:
        # Pre-compute the Kernel Matrix
        # The following can be slow due to the lack of vectorization
        K = np.zeros([m,m])
        for i in range(m):
            for j in range(m):
                 K[i,j] = kernelFunction(X[i,:], X[j,:],sigma)
                 K[j,i] = K[i,j]
      
    # Train 
    print('Training:')
    dots = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # Calculate Ei = f(x(i)) - y(i) using (2)
            # E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
            E[i] = b + np.sum(alphas[:,0]*Y[:,0]*K[:,i]) - Y[i,0]
        
            if ((Y[i,0]*E[i,0] < -tol and alphas[i,0] < C) or (Y[i,0]*E[i,0] > tol and alphas[i,0] > 0)):
                j = int(np.floor(m * np.random.rand()))
                while j == i:
                    j = int(np.floor(m * np.random.rand()))# Make sure i \neq j            

                # Calculate Ej = f(x(j)) - y(j)
                E[j] = b + np.sum(alphas[:,0]*Y[:,0]*K[:,j]) - Y[j,0]

                # Save old alphas
                alpha_i_old = alphas[i,0]
                alpha_j_old = alphas[j,0]
            
                # Compute L and H   by (10) or (11)
                if (Y[i,0] == Y[j,0]):
                    L = np.max([0, alphas[j,0] + alphas[i,0] - C])
                    H = np.min([C, alphas[j,0] + alphas[i,0]])
                else:
                    L = np.max([0, alphas[j,0] - alphas[i,0]])
                    H = np.min([C, C + alphas[j,0] - alphas[i,0]])
            
           
                if (L == H):
                    # continue to next i. 
                    continue
        

                # Compute eta by (14)
                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if (eta >= 0):
                    # continue to next i. 
                    continue
        
        
                # Compute and clip new value for alpha j using (12) and (15)
                alphas[j,0] = alphas[j,0] - Y[j,0]*(E[i,0] - E[j,0]) / eta
            
                # Clip
                alphas[j,0] = np.min([H, alphas[j,0]])
                alphas[j,0] = np.max([L, alphas[j,0]])
            
                # Check if change in alpha is significant
                if (abs(alphas[j,0] - alpha_j_old) < tol):
                    # continue to next i. 
                    # replace anyway
                    alphas[j,0] = alpha_j_old
                    continue
                
            
                # Determine value for alpha i using (16)
                alphas[i,0] = alphas[i,0] + Y[i,0]*Y[j,0]*(alpha_j_old - alphas[j,0])
            
                # Compute b1 and b2 using (17) and (18) respectively
                b1 = b - E[i,0] \
                     - Y[i,0] * (alphas[i,0] - alpha_i_old) *  K[i,i] \
                     - Y[j,0] * (alphas[j,0] - alpha_j_old) *  K[i,j]
                b2 = b - E[j,0] \
                     - Y[i,0] * (alphas[i,0] - alpha_i_old) *  K[i,j] \
                     - Y[j,0] * (alphas[j,0] - alpha_j_old) *  K[j,j]

                # Compute b by (19)
                if (0 < alphas[i,0] and alphas[i,0] < C):
                    b = b1
                elif (0 < alphas[j,0] and alphas[j,0] < C):
                    b = b2
                else:
                    b = (b1+b2)/2
            

                num_changed_alphas = num_changed_alphas + 1

        
    
        if (num_changed_alphas == 0):
            passes = passes + 1
        else:
            passes = 0
    

        sys.stdout.write('.')
        dots = dots + 1
        if dots > 78:
            dots = 0
            sys.stdout.write('\n')
            
    print('Done!',)
    # Save the model
    model = dict()
    idx = alphas[:,0] > 0
    model['X']= X[idx,:]
    model['y']= Y[idx,0]
    model['kernelFunction'] = kernelFunction
    model['b']= b
    model['alphas']= alphas[idx,0]
    model['w'] = ((alphas*Y).T.dot(X)).T
    model['sigma'] = sigma
    return model    
    
def svmPredict(model, X):
    #SVMPREDICT returns a vector of predictions using a trained SVM model
    #(svmTrain). 
    
    #Check if we are getting a column vector, if so, then assume that we only
    # need to do prediction for a single example
    if (X.shape[1] == 1):
        # Examples should be in rows
        X = X.T

    # Dataset 
    m = X.shape[0]
    p = np.zeros([m, 1])
    pred = np.zeros([m, 1])

    if model['kernelFunction'].__name__ == 'linearKernel':
        # We can use the weights and bias directly if working with the 
        # linear kernel
        p = X.dot(model['w']) + model['b']
    elif model['kernelFunction'].__name__ == 'gaussianKernel':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X1 = np.sum(X**2, 1).reshape([-1,1])
        X2 = np.sum(model['X']**2, 1).reshape([1,-1])
        K = X2 - 2 * X.dot(model['X'].T)
        K =  X1 + K
        K = model['kernelFunction'](1, 0,model['sigma'])** K
          
        K = model['y'].T * K
        K = model['alphas'].T * K
        p = np.sum(K, 1)
    else:
        # Other Non-linear kernel
        for i in range(m):
            prediction = 0
            for j in range(model['X'].shape[0]):
                prediction = prediction + \
                    model['alphas'][j] * model['y'][j] * \
                    model['kernelFunction'](X[i,:], model['X'][j,:])
            
            p[i] = prediction + model['b']


    # Convert predictions into 0 / 1
    #pred[p[:,0] >= 0,0] =  1
    #pred[p[:,0] <  0,0] =  0
    pred = np.where(p>=0,1,0)
    return pred


def visualizeBoundary(X, y, model):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    #   boundary learned by the SVM and overlays the data on it

    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).T
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros_like(X1)
    for i in range(X1.shape[1]):
       this_X = np.hstack([X1[:, i].reshape([-1,1]), X2[:, i].reshape([-1,1])])
       vals[:, i] = svmPredict(model, this_X)

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, levels=[0], color ='b')

def processEmail(email_contents):
    #PROCESSEMAIL preprocesses a the body of an email and
    #returns a list of word_indices 

    # Load Vocabulary
    vocabList = pd.read_csv('vocab.txt',sep='\t',header=-1)[1].to_dict()
    vocab_size= len(vocabList)
    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Lower case
    email_contents = email_contents.lower()
    # remove all '\n'
    email_contents = re.sub(r'\n','',email_contents)
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+','number',email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr',email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr',email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar',email_contents)


    # ========================== Tokenize Email ===========================
    # Output the email to screen as well
    # Process file
    
    # Tokenize and also get rid of any punctuation
    words = nltk.tokenize.word_tokenize(email_contents)
    punctuations = ' @$/#.-:&*+=[]?!(){},''">_<;%'
    words = [w for w in words if w not in punctuations]
   
    # Remove any non alphanumeric characters
    words = [re.sub('[^a-zA-Z0-9]', '',w)  for w in words]

    # Stem the word 
    # (the porterStemmer sometimes has issues, so we use a try catch block)
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    
    # Look up the word in the dictionary and add to word_indices if found

    word_indices = [ind for w in words for ind in vocabList.keys() if w == vocabList[ind] ]
    return word_indices, words

def emailFeatures(word_indices):
    #EMAILFEATURES takes in a word_indices vector and produces a feature vector
    #from the word indices

    # Total number of words in the dictionary
    n = 1899
    x = np.zeros([n, 1])
    x[word_indices,0] = 1
    return x