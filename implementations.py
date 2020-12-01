# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np

import matplotlib.pyplot as plt

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=y-tx.dot(w)
    gradient=(-1/y.shape[0])*np.transpose(tx).dot(e)
    return gradient

def compute_loss(y, tx, w):
    """Calculate the loss."""
    e=y-tx.dot(w)
    loss=(1/(2*y.shape[0]))*(np.transpose(e).dot(e))
    return loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=y-tx.dot(w)
    if len(y)>1:
        stoch_gradient=(-1/y.shape[0])*np.transpose(tx).dot(e)
    else:
        stoch_gradient=-tx*e
    return stoch_gradient


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def sigmoid(t):
    """apply sigmoid function on t."""
    sigmoid=1/(1+np.exp(-t))
    return sigmoid

def calculate_loss2(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss=0
    for i in range(len(y)):
        loss=loss+(np.log(1+np.exp(np.transpose(tx[i,:]).dot(w)[0]))-y[i][0]*np.transpose(tx[i,:]).dot(w)[0])
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    gradient=np.dot(np.transpose(tx),sigmoid(np.dot(tx,w))-y)
    return gradient

def cross_validation(y, x,seed):
    """return the loss of ridge regression."""

    k_fold = 4
    lambdas = np.logspace(-5, 0, 50)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
   
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for ind,lambda_ in enumerate(lambdas):
        loss_tr=[]
        loss_te=[]
        for k in range(4):
            y_test=y[k_indices[k]]
            x_test=x[k_indices[k],:]
            
            x_train=np.zeros((k_indices.shape[1]*(k_fold-1),x.shape[1]))
            y_train=[]
            c=0
            d=1
            for i in range(len(k_indices)):
                
                if i!=k:
                    x_train[c*k_indices.shape[1]:d*k_indices.shape[1],:]=x[k_indices[i],:]
                    y_train=np.append(y_train,y[k_indices[i]])
                    c+=1
                    d+=1
            
            w,loss_tr1=ridge_regression(y_train, x_train, lambda_)
            loss_te1=compute_loss(y_test,x_test,w)
            loss_tr.append(np.sqrt(2*loss_tr1))
            loss_te.append(np.sqrt(2*loss_te1))
        rmse_tr.append(np.mean(loss_tr))
        rmse_te.append(np.mean(loss_te))   
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    e=np.argmin(rmse_te)
    return lambdas[e]

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")    
    
"""functions"""    
    
def least_squares_GD(y,tx,initial_w,max_iters,gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        w=w-gamma*gradient
        loss=compute_loss(y,tx,w)
        
    return w,loss

def least_squares_SGD(y,tx,initial_w,max_iters,gamma,batch_size=1):
    """Stochastic gradient descent algorithm."""
    
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            gradient=compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            loss=compute_loss(y,tx,w)
            w=w-gamma*gradient
         
    return w,loss  


def least_squares(y, tx):
    """calculate the least squares solution."""
    if np.linalg.det(np.transpose(tx).dot(tx))!=0:
        w=(np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx))).dot(y)
        e=y-tx.dot(w)
        loss=(1/(2*y.shape[0]))*(np.transpose(e).dot(e))
    else:
        w=np.linalg.solve(tx,y) 
        loss=(1/(2*y.shape[0]))*(np.transpose(e).dot(e))
    return w,loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    I=np.ones((tx.shape[1],tx.shape[1]))
    lambda_1=lambda_*2*len(y)
    w=np.dot(np.linalg.inv(np.dot(tx.T,tx)+lambda_1*I),np.dot(tx.T,y))
    e=y-tx.dot(w)
    loss=(1/(2*y.shape[0]))*(np.transpose(e).dot(e))
    
    return w,loss



def logistic_regression(y, tx, initial_w,max_iters,gamma):
    
    threshold = 1e-3
    losses=[]
    w=initial_w
    for iter in range(max_iters):
        # get loss and update w.
        loss=calculate_loss2(y, tx, w)
        gradient=calculate_gradient(y, tx, w)
        w=w-gamma*gradient
        
        losses.append(loss)
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]) < threshold ):
            break    
    return  w,loss

def reg_logistic_regression(y, tx,lambda_, initial_w,max_iters,gamma):
    
    gamma=1.2e-6
    max_iter = 1500
    threshold = 1e-3
    w=initial_w
    for iter in range(max_iter):
        # get loss and update w.
        loss=calculate_loss2(y, tx, w)+lambda_*np.linalg.norm(w,2)
        gradient=calculate_gradient(y, tx, w)+2*lambda_*w
        w=w-gamma*gradient
        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]) < threshold ):
            break    
    return w,loss

def standardize(x):
    """Standardize the original data set."""
    min_ = np.amin(x,0)
    max_=np.amax(x,0)
    for i in range(x.shape[1]):
        x[:,i]=x[:,i]-min_[i]
        diff=max_[i]-min_[i]
        x[:,i]=x[:,i]/diff
    return x

def batch(y, tx, batch_size,seed):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    np.random.seed(seed)
    data_size=len(y)
    indices = np.random.permutation(tx.shape[0])
    batch_ind=indices[:batch_size]
    minibatch_y=y[batch_ind]
    minibatch_x=tx[batch_ind]
    return minibatch_y,minibatch_x