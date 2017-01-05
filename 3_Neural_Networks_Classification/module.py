import scipy.io as sio # for loading matlab files
import numpy as np
import scipy.optimize as op

import matplotlib.pyplot as plt
def displayData(X,example_width=None):
    if example_width is None:
        example_width = int(np.sqrt(X.shape[1]))
    m,n = X.shape
    example_height = int(n/example_width)
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    # Between images padding
    fig,ax = plt.subplots(nrows=display_rows,ncols=display_cols)
    fig.set_size_inches(10,10)
    #fig.tight_layout()
    k=0
    for j in range(display_rows):
        for i in range(display_cols):
            ax[j,i].axis('off')
            ax[j,i].imshow(X[k,:].reshape([example_height,example_width]).T,cmap=plt.cm.binary)
            k+=1
            
def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    return 1.0/ (1.0 + np.exp(-z))

def grad(theta,X,y,lam):
    m,n = X.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    h = sigmoid(X.dot(theta))
    grad = X.T.dot(h-y)/m
    grad[1:] = grad[1:] + lam*theta[1:]/m
    return grad.flatten()

def costFun(theta, X, y,lam):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    # J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t. to the parameters.
    m,n = X.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))

    h = sigmoid(X.dot(theta))
    J = (-y.T.dot(np.log(h))-(1-y.T).dot(np.log(1-h)))/m
    J = J + lam*np.sum(theta[1:]**2)/2/m;

    return J.flatten()

def oneVsAll(X,y,num_labels,lam):
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta 
    #corresponds to the classifier for label i
    m,n = X.shape
    # Add ones to the X data matrix
    X1 = np.vstack([np.ones(m), X.T]).T
    all_theta = np.zeros((num_labels,n+1))
    init_theta = np.zeros((n+1,1))
    for c in range(num_labels):
        theta = op.fmin_cg(f=costFun,args=(X1,np.where(y==c,1,0),lam),
                     x0=init_theta,fprime=grad,disp=0)
        all_theta[c,:] = theta
    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    # add ones to X
    X1 = np.vstack([np.ones(m), X.T]).T
    p = np.argmax(sigmoid(X1.dot(all_theta.T)),axis=1)
    return p.reshape([-1,1])

    
def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = np.vstack([np.ones(m), X.T]).T
    a2 = sigmoid(a1.dot(Theta1.T))
    a2 = np.vstack([np.ones(a2.shape[0]), a2.T]).T
    a3 = sigmoid(a2.dot(Theta2.T))
    p = np.argmax(a3,axis=1)+1
    indx = p ==10; p[indx]=0
    return p.reshape([-1,1])