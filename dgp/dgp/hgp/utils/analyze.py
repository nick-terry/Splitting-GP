# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:15:28 2015

@author: marc
"""

import numpy as np

# negative log predictive density
def nlpd(mpred, Spred, ytrue):
    maha = (mpred - ytrue)**2/Spred/2
    logDet = np.log(Spred)
    NLPD = logDet + maha + np.log(np.pi*2.0)/2
    return NLPD
  
# mean square error  
def mse(y, ypred):
  sq_d = (y - ypred)**2
  return np.mean(sq_d)
  
# normalized mean squared error
def nmse(y, ypred):
  return mse(y,ypred)/(np.mean(y)*np.mean(ypred))
  
# mean absolute error
def mae(y, ypred):
  return np.mean(np.abs(y-ypred))

# root mean squared error
def rmse(y, ypred): 
  return np.sqrt(mse(y, ypred))

# normalized root mean squared error  
def nrmse(y, ypred):
  return rmse(y, ypred)/(np.max(ypred)-np.min(ypred))
    
# normalize data (mean 0 , variance 1)    
def whitenData(x,xmean=None,xstd=None):
    # assume x is form NxD
    # xWhitened = (x - xmean)/xstd
    if xmean==None:
        xmean=np.mean(x,axis=0)
        xstd=np.std(x,axis=0)
        xWhitened = (x-xmean)/xstd
        return xWhitened, xmean, xstd
    
    else:
        xWhitened = (x-xmean)/xstd
        return xWhitened
        
# undo the normalization
def unWhitenData(xWhitened,xmean,xstd):
    # assume x is form NxD
    # x = xWhitened*xstd + xean
    x = xWhitened*xstd + xmean    
    return x

# transform (Gaussian) predictions of normalized variables into predictions in original space
def unWhitenPrediction(m, S, xmean, xstd):
    # un-do the whitening in the prediction
    # x = xWhitened*xstd + xmean
    # where xWhitened ~ N(x|m, S)
    # then x ~ N(x| m*xstd, xstd^2*S)
    mu = m*xstd + xmean
    Sigma = S*xstd**2
    return mu, Sigma
    