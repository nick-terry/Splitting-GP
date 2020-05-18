# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:09:11 2015

@author: marc
"""

import sys
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import cPickle as pk
import string
from hgp import HGP
from hgp.utils.tictoc import *

  
numbers = []
f=open('data/kin-32nh/Dataset.data','r')
for eachLine in f:
    eachLine = eachLine.strip()
    y = [float(value) for value in eachLine.split()]
    numbers.append(y)   
f.close()  
    

data = np.asarray(numbers)

N,d = data.shape


X = data[:,0:d-2]
y = data[:,d-1]


# i_t, i_s - training, test data indices from full data set
r = range(1, N)
shuffle(r)

train_N = np.int(np.ceil(N/2))
test_N = np.int(N - train_N)

i_t = r[:train_N]
i_s = r[train_N:N]

Xtrain = X[i_t,:]
ytrain = y[i_t]

Xtest = X[i_s,:]
ytest = y[i_s]

profile = [(4,'random','rep2')]*5

results = {}

print 'Setting up the model...'
tic()
hgp1 = HGP(Xtrain,ytrain, profile=profile, pool = 'default')
time = toc()
print 

print 'Training..'
tic()
hgp1.train()
time = toc()

print 'Predicting..'
ymu, ys2 = hgp1.predict(Xtest, variance=True)
smse = np.mean((ymu-ytest)**2)/np.var(ytest)
print 'SMSE: %f' % smse

#results['bfgs_niter'] = hgp1.lik_count
results['train_time'] = time
results['hgp_params'] = hgp1.params
results['smse'] = smse
results['ytest'] = ytest
results['ymu'] = ymu
results['ys2'] = ys2
results['i_t'] = i_t
results['i_s'] = i_s
