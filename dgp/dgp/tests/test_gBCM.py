# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 08:16:00 2015

@author: marc
"""

#import copy

import numpy as np
import matplotlib.pyplot as plt

#from hgp import HGP
#from hgp import BCM
#from hgp import GP
from hgp import gBCM
# from hgp import gPoE



N = 1000 # no of training inputs
d = 1  # no of input dimensions

np.random.seed(1)

# training data
X = np.random.uniform(-4,4,(N,d))
y = np.sin(X.sum(axis=1)) + np.random.normal(0,0.3,N)

# test data
if d == 1:
    Xp = np.linspace(-10,10,200).reshape(-1,1)
else:
    Xp = np.random.uniform(-4,4,(4000,d))
    
#gp1 = GP(X,y)
#gp1.params = [0,0,np.log(0.01)]
#gp1.train()
#mugp, s2gp= gp1.predict(Xp,variance=True) # OK


profile=[(4,'simple','rep1')]*4
gbcm0 = gBCM.gBCM(X,y,profile=profile, pool='default')
gbcm0.correction = True
#ymu1, s1 = gbcm0.predict(Xp,latent_variance=True)
#print 'NLML before training',
#print gbcm0.NLML()

print 'training HGP...'
gbcm0.train()
print 'NLML after training',
print gbcm0.NLML()

#gbcm0.beta=1.0
# prediction means and variances
ymu, ys22 = gbcm0.predict(Xp,latent_variance=True)
ys22 += np.exp(gbcm0.params[-1])

## 2sd bands
y2sd2 = 2*np.sqrt(ys22)
#
#
#gp1.params = gbcm0.params
#mugp, s2gp= gp1.predict(Xp,variance=True) # OK
#
## HGP prediction
#
#hgp=HGP(X,y,profile=profile, pool='default')
#hgp.params = gbcm0.params
#mh, sh = hgp.predict(Xp, latent_variance=True)
#sh += np.exp(gbcm0.params[-1])
if d == 1:
    # plot data if it's 1d
    plt.scatter(X,y)
    #plt.plot(Xp,ymu1,color='b')
    plt.plot(Xp,ymu,color='c')
    plt.plot(Xp,ymu+y2sd2,color='c')
    plt.plot(Xp,ymu-y2sd2,color='c')
    plt.show()
#    
#    plt.plot(Xp, mugp, color='c')
#    plt.plot(Xp, mugp -2*np.sqrt(s2gp), color='c')
#    plt.plot(Xp, mugp +2*np.sqrt(s2gp), color='c')
##    
##    plt.plot(Xp,mh,color='g')
##    plt.plot(Xp,mh+2.0*np.sqrt(sh),color='g')
##    plt.plot(Xp,mh-2.0*np.sqrt(sh),color='g')  
##    plt.show()
#
