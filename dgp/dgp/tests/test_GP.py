import numpy as np
import matplotlib.pyplot as plt

from dgp import GP

N = 200 # no of training inputs
d = 2 # no of input dimensions

# training data
X = np.random.uniform(-4,4,(N,d))
y = np.sin(X.sum(axis=1)) + np.random.normal(0,0.3,N)

# test data
if d == 1:
    Xp = np.linspace(-4,4,100).reshape(-1,1)
else:
    Xp = np.random.uniform(-4,4,(100,d))

gp0 = GP(X,y,cov_type='covSEard') # create GP with default cov

ymu1 = gp0.predict(Xp)
print 'NLML before training',
print gp0.NLML()

print 'training GP..'
gp0.train()
print 'NLML after training',
print gp0.NLML()

# prediction means and variances
ymu2, ys22 = gp0.predict(Xp,variance=True)

# 2sd bands
y2sd2 = np.sqrt(4.0*ys22)

if d == 1:
    # plot data if it's 1d
    plt.scatter(X,y)
    plt.plot(Xp,ymu1,color='b')
    plt.plot(Xp,ymu2,color='r')
    plt.plot(Xp,ymu2+y2sd2,color='r')
    plt.plot(Xp,ymu2-y2sd2,color='r')
    plt.show()
