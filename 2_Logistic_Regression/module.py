import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    return 1/(1+np.exp(-z))


def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure 
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.
    ind1 = np.where(y==1)[0]
    plt.plot(X[ind1,0],X[ind1,1],'k+',linewidth=2, markersize= 7,label='Admitted')
    ind2 = np.where(y==0)[0]
    plt.plot(X[ind2,0],X[ind2,1],'ko',markerfacecolor = 'y',markersize= 7,label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()

    
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

def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of 
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size


    degree = 6
    if type(X1)==np.ndarray:
        n =  len(X1)
        out = np.ones([1,n])
    else:
        n=X1
        out=1
    for i in range(1,degree+1):
        for j in range(i+1):
            out =  np.vstack([out, ((X1**(i-j))*(X2**(j))).T])
    
    return out.T

def plotDecisionBoundary(theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    #   positive examples and o for the negative examples. X is assumed to be 
    #   a either 
    #   1) Mx3 matrix, where the first column is an all-ones column for the 
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    plotData(X[:,[1,2]], y)
    ax =  plt.gca()

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,2])+2])
    
        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
    
        # Plot, and adjust axes for better viewing
        ax.plot(plot_x, plot_y)
    
        # Legend, specific for the exercise
        ax.legend(('Admitted', 'Not admitted', 'Decision Boundary'))
        ax.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros([len(u), len(v)])
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = mapFeature(u[i], v[j]).dot(theta)
           
        z = z.T # important to transpose z before calling contour

        # Plot z = 0
        ax.contour(u, v, z, levels=[0], linewidth = 2)




def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic 
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a 
    #   threshold at 0.5 (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    n,m = X.shape # Number of training examples
    p = np.zeros([n,1])
    ind1 =np.where(sigmoid(X.dot(theta)) >= 0.5)
    p[ind1] = 1
    return p

