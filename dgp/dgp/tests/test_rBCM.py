# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 08:16:00 2015

@author: marc
"""



import numpy as np
import matplotlib.pyplot as plt

from dgp import DGP
from dgp import BCM
from dgp import GP
from dgp import rBCM
from dgp import gPoE
from dgp.utils import tictoc


N = 1000 # no of training inputs
d = 1 # no of input dimensions

np.random.seed(1)

# training data
# training data
X = np.random.uniform(-4,4,(N,d))
y = np.sin(X.sum(axis=1)) + np.random.normal(0,0.1,N)

# test data
Xp = np.linspace(-8,8,200).reshape(-1,1)

branching_factor = 12
depth = 1
numExperts = branching_factor**depth
# repX stands for X-fold repetition of data points. This implements shared data 
# points across experts. "X" is the number of times a data point is "replicated" 
profile=[(branching_factor,'simple','rep1')]*depth
dgp0 = rBCM.rBCM(X,y,profile=profile, pool='default')

print 'training DGP...'
tictoc.tic()
dgp0.train()
tictoc.toc()

print 'NLML after training',
print dgp0.NLML()




# prediction means and variances
print 'Predicting...'

# rBCM
dgp0.correction = True
if hasattr(dgp0, 'beta'):
  del dgp0.beta
meanPred_rbcm, varPred_rbcm= dgp0.predict(Xp,latent_variance=True)
varPred_rbcm += np.exp(dgp0.params[-1])
# gPoE
dgp0.correction = False  # no BCM-type correction (prior GP)
dgp0.beta = 1./numExperts # fix beta values
meanPred_gpoe, varPred_gpoe= dgp0.predict(Xp,latent_variance=True)
varPred_gpoe += np.exp(dgp0.params[-1])
# PoE
dgp0.correction = False # no BCM-type correction (prior GP)
dgp0.beta = 1.0 # fix beta values
meanPred_poe, varPred_poe= dgp0.predict(Xp,latent_variance=True)
varPred_poe += np.exp(dgp0.params[-1])
# BCM
dgp0.correction = True
dgp0.beta = 1.0 # fix beta values
meanPred_bcm, varPred_bcm= dgp0.predict(Xp,latent_variance=True)
varPred_bcm += np.exp(dgp0.params[-1])

if d == 1:
    # rBCM
    plt.scatter(X,y)
    plt.plot(Xp,meanPred_rbcm,color='r')
    plt.plot(Xp,meanPred_rbcm+2.0*np.sqrt(varPred_rbcm),color='r')
    plt.plot(Xp,meanPred_rbcm-2.0*np.sqrt(varPred_rbcm),color='r')
    plt.show()
    
    # PoE
    plt.scatter(X,y)
    plt.plot(Xp,meanPred_poe,color='b')
    plt.plot(Xp,meanPred_poe+2.0*np.sqrt(varPred_poe),color='b')
    plt.plot(Xp,meanPred_poe-2.0*np.sqrt(varPred_poe),color='b')
    plt.show()

    # gPoE
    plt.scatter(X,y)
    plt.plot(Xp,meanPred_gpoe,color='g')
    plt.plot(Xp,meanPred_gpoe+2.0*np.sqrt(varPred_gpoe),color='g')
    plt.plot(Xp,meanPred_gpoe-2.0*np.sqrt(varPred_gpoe),color='g')
    plt.show()
    
    # BCM
    plt.scatter(X,y)
    plt.plot(Xp,meanPred_bcm,color='m')
    plt.plot(Xp,meanPred_bcm+2.0*np.sqrt(varPred_bcm),color='m')
    plt.plot(Xp,meanPred_bcm-2.0*np.sqrt(varPred_bcm),color='m')
    plt.show()
    
    