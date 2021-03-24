import sys

# from hgp.utils import getdata as gd
from hgp import DataSet as DS
from hgp import HGP
from hgp import GP
from hgp.utils.tictoc import *
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import cPickle as pk
import string
import hgp

#def usage(message=None):
#    if message is not None:
#        print '\n'+message
#    print 'usage: python experiment.py <architecture> <training size> <test size>'
#    print 'example: python experiment.py 4,4,4 1000000 100000\n'
#    raise Exception
#
#try:
#    arch = [ int(x) for x in string.split(sys.argv[1],',') ]
#except:
#    usage('no HGP architecture specified')
#
#try:
#    train_N = int(sys.argv[2])
#except:
#    usage('no training set size specified')
#
#try:
#    test_N = int(sys.argv[3])
#except:
#    usage('no test set size specified')

with open('experiments/filtered_data.pickle','r') as f:
    data = pk.load(f)

y = data['ArrDelay']
X = data[['Month','DayofMonth','DayOfWeek','DepTime','ArrTime','AirTime','Distance','plane_age']]

# hgp0 = hgp.HGP(X,y,pool='default',profile=[(100,'random','rep2')]*3) # create GP with default cov

data = hgp.DataSet.DataSet(X,y)

# i_t, i_s - training, test data indices from full data set
r = range(1,len(y))
shuffle(r)

train_N = 700000
test_N = 100000

i_t = r[:train_N]
i_s = r[train_N:(train_N+test_N)]

# Xt, yt - training data
# Xs, ys - test data
#Xt, yt = data.subset(i_t).data()
#Xs, ys = data.subset(i_s).data()

Xt = np.asarray(data.X)[i_t,:]
yt = data.y[i_t]


Xs = np.asarray(data.X)[i_s,:]
ys = data.y[i_s]

# profile = [ (a,'rep2','kd_cluster') for a in arch ]
profile = [(50,'random','rep2')]*3
results = {}

hgp1 = hgp.HGP.HGP(Xt,yt,profile=profile)

print 'Training..'
tic()
hgp1.train()
time = toc()

print 'Predicting..'
ymu, ys2 = hgp1.predict(Xs,variance=True)
smse = np.mean((ymu-ys)**2)/np.var(ys)
print 'SMSE: %f' % smse

results['bfgs_niter'] = hgp1.lik_count
results['train_time'] = time
results['hgp_params'] = hgp1.params
results['smse'] = smse
results['ys'] = ys
results['ymu'] = ymu
results['ys2'] = ys2
results['i_t'] = i_t
results['i_s'] = i_s


filename = 'results_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'.pk'

with open(filename,'wb') as f:
    pk.dump(results,f)

print 'architecture '+sys.argv[1]+' done'
print 'results saved in '+filename


