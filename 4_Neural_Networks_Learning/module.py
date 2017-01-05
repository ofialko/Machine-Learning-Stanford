import scipy.io as sio # for loading matlab files
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from scipy.io import loadmat

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
    return 1.0/(1.0 + np.exp(-z))

def sigmoidGradient(z):
    #SIGMOIDGRADIENT returns the gradient of the sigmoid function
    #evaluated at z

    sig = sigmoid(z)
    g = sig*(1-sig)
    return g

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,X, y, lam):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, 
    # the weight matrices for our 2 layer neural network
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                       [hidden_layer_size,(input_layer_size+1)])
    
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
                       [num_labels,(hidden_layer_size+1)])
    m = len(y)
    
    # a1 is m by 1+input_layer_size
    #a1 = np.vstack([np.ones(m), X.T]).T
    a1 = np.hstack([np.ones(m).reshape([-1,1]),X])
    # z2 is m by hidden_layer_size
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    # a2 is m by 1 + hidden_layer_size
    #a2 = np.vstack([np.ones(a2.shape[0]), a2.T]).T
    a2 = np.hstack([np.ones(a2.shape[0]).reshape([-1,1]),a2])
    # a3 is m by num_labels
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    
    y1 = y.squeeze()
    ym = np.zeros([m,num_labels])
    for k in range(num_labels):
        ym[:,k] = np.where(y1==k,1,0)
        
    #ym=np.roll(ym, -1, axis=1)
    
    J = np.sum(-ym*np.log(a3)-(1.0-ym)*np.log(1.0-a3))/m
    J = J +(np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))*lam/2/m
    
    delta3 = a3 - ym
    delta2 = delta3.dot(theta2)*a2*(1-a2)
    delta2 = delta2[:,1:]
    
    theta1_grad = delta2.T.dot(a1)/m
    theta2_grad = delta3.T.dot(a2)/m

    theta1_grad[:,1:] = theta1_grad[:,1:] + theta1[:,1:]*lam/m
    theta2_grad[:,1:] = theta2_grad[:,1:] + theta2[:,1:]*lam/m

    # Unroll gradients
    grad = np.hstack([theta1_grad.ravel(), theta2_grad.ravel()])#.reshape([-1,1])
    return (J,grad)

def randInitializeWeights(L_in, L_out):
    #RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    #incoming connections and L_out outgoing connections

    epsilon_init = 0.12

    W = np.random.rand(L_out, L_in+1) * 2 * epsilon_init - epsilon_init
    return W

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
    p = np.argmax(a3,axis=1)#+1
    #indx = p ==10; p[indx]=0
    return p.reshape([-1,1])

