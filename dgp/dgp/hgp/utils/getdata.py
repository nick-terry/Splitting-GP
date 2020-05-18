import os
import cPickle as pk

import numpy as np
import pandas as pds

def fromFile(filepath):
    '''
    reads data from csv file 
    '''
    data = pds.read_csv(filepath,header=None)
    return data.values

def sample_data(id):
    current = os.path.dirname(__file__)
    dir = current + '/../../data/sample' + str(id)
    X = fromFile(dir+'/train_inputs')
    y = fromFile(dir+'/train_outputs').squeeze()
    return X,y

def generate1d(description='sin',N=200,seed=None,
               cov_type='covSEiso',cov_params=None,randx=False):
    np.random.seed(seed)
    if randx:
        n, r = divmod(N,2) 
        X1 = np.random.normal(-3,2,n).flatten().squeeze()
        X2 = np.random.normal(3,2,n+r).flatten().squeeze()
        X = np.hstack([X1,X2])
    else:
        X = np.linspace(0,5,N)

    if description == 'sin':
        y = np.sin(X)
        y += np.random.normal(0,np.std(y)*0.75,N)
        X.shape = -1,1
    elif description == 'sin2':
        y = np.sin(X) + np.sin(2*X)
        y += np.random.normal(0,np.std(y)*0.75,N)
        X.shape = -1,1
    elif description == 'sin3':
        y = np.sin(X) + np.sin(2*X) + np.sin(3*X)
        y += np.random.normal(0,np.std(y)*0.75,N)
        X.shape = -1,1
    elif description == 'kernelpaper':
        X1 =  np.random.uniform(-10,10,N)
        X2 =  np.random.uniform(-10,10,N)
        X1.shape = -1,1
        X2.shape = -1,1
        X = np.hstack([X1, X2])
   
        normX = np.sqrt(np.sum(X*X,axis=1))
 
        y = np.cos(np.pi/2*normX)*np.exp(-0.1*np.pi*normX)
        y += np.random.normal(0,0.1,N)
   
    elif description == 'gp':
        X.shape = -1,1
        if cov_params is None:
            cov_params = np.asarray([0,0,0])
        k = GPcov.cov(cov_type,cov_params[:-1],X,X)
        k += np.eye(N)*np.exp(cov_params[-1]) 
        y = np.random.multivariate_normal(np.zeros(N),k,1).squeeze()
    elif description == 'snelson':
        X,y = sample_data(1)

    return X,y
