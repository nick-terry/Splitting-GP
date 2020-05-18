import copy

import numpy as np
import matplotlib.pyplot as plt

from hgp import HGP
from hgp.utils import tictoc


N = 512 # no of training inputs
d = 1 # no of input dimensions

np.random.seed(1)

# training data
X = np.random.uniform(-4,4,(N,d))
y = np.sin(X.sum(axis=1)) + np.random.normal(0,0.3,N)

# test data
if d == 1:
    Xp = np.linspace(-8,8,200).reshape(-1,1)
else:
    Xp = np.random.uniform(-4,4,(100,d))
    
    
profile=[(2,'simple','rep1')]*4
hgp0 = HGP(X,y,profile=profile, pool='default') # create GP with default cov

#ymu1 = hgp0.predict(Xp)
print 'NLML before training',
print hgp0.NLML()

print 'training HGP...'
tictoc.tic()
hgp0.train()
tictoc.toc()
print 'NLML after training',
print hgp0.NLML()

# prediction means and variances
print 'Predicting...'
tictoc.tic()
ymu2, ys22 = hgp0.predict(Xp,variance=True)
tictoc.toc()
# 2sd bands
y2sd2 = np.sqrt(4.0*ys22)

if d == 1:
    # plot data if it's 1d
    plt.scatter(X,y)
    #plt.plot(Xp,ymu1,color='b')
    plt.plot(Xp,ymu2,color='r')
    plt.plot(Xp,ymu2+y2sd2,color='r')
    plt.plot(Xp,ymu2-y2sd2,color='r')
    plt.show()
