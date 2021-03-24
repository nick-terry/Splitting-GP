import numpy as np

from hgp import DataSet as DS

N = 2**10
d = 20

X = np.random.normal(0,1,(N,d))
y = np.random.normal(0,1,N)

d = DS(X,y)
subsets = d.partition(16,'kd_uniform')
