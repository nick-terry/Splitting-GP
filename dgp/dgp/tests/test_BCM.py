import copy

import numpy as np
import matplotlib.pyplot as plt

from hgp import HGP
from hgp import BCM
from hgp import GP



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
    
#gp1 = GP(X,y)
#gp1.params = [0,0,np.log(0.01)]
#mugp, s2gp= gp1.predict(Xp,variance=True) # OK


profile=[(4,'simple','rep1')]*4
#bcm0.params = [0,0,np.log(0.01)]
bcm0 = BCM.BCM(X,y,profile=profile) # create GP with default cov

#ymu1, s1 = bcm0.predict(Xp,latent_variance=True)
print 'NLML before training',
print bcm0.NLML()

print 'training HGP...'
bcm0.train()
print 'NLML after training',
print bcm0.NLML()

# prediction means and variances
ymu2, ys22 = bcm0.predict(Xp,latent_variance=True)
ys22 += np.exp(bcm0.params[-1])


hgp0 = HGP(X,y,profile=profile, pool='default') 
hgp0.train()
m1, s1 = hgp0.predict(Xp, latent_variance=True)
s1 += np.exp(hgp0.params[-1])

# 2sd bands
y2sd2 = 2*np.sqrt(ys22)

if d == 1:
    # plot data if it's 1d
    plt.scatter(X,y)
    #plt.plot(Xp,ymu1,color='b')
    plt.plot(Xp,ymu2,color='c')
    plt.plot(Xp,ymu2+y2sd2,color='c')
    plt.plot(Xp,ymu2-y2sd2,color='c')
    plt.show()
    
    plt.scatter(X,y)
    #plt.plot(Xp,ymu1,color='b')
    plt.plot(Xp,m1,color='g')
    plt.plot(Xp,m1+2*np.sqrt(s1),color='g')
    plt.plot(Xp,m1-2*np.sqrt(s1),color='g')
    plt.show()
