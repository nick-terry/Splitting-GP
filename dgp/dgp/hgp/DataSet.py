import pdb

import copy
from random import shuffle

import numpy as np


class DataSet(object):

    def __init__(self,X=None,y=None,superset=None,indices=None):

        # a DataSet object contains pairs of inputs (in X)
        # and outputs (in y). It can be a subset of another
        # DataSet object, in which case its data will be described
        # by an array of indices referencing the position of its data
        # in the superset's arrays

        self._X = X
        self._y = y

        self._superset = superset
        self._indices = np.asarray(indices)

    def subset(self,indices):
        # returns a DataSet object which contains a subset of the data
        I = np.asarray(indices)
        subDS = DataSet(None,None,superset=self,indices=I)

        return subDS

    def add(self,other):

        # add another DataSet object to the current one

        if self.is_subset: self.resolve()
        if other.is_subset: other.resolve()

        if self.superset is not other.superset:
            self.detach()
            self._X = np.vstack([self.X,other.X])
            self._y = np.hstack([self.y,other.y])
        else:
            self._indices = np.hstack([self.indices,other.indices])

    def cache(self):
        if self.is_subset:
            self._X, self._y = self.X, self.y

    def partition(self,k=2,method='simple',**kwargs):

        # partitions the data set into a list of non overlapping ones
        # k = number of resulting partitions

        if method == 'random' or method == 'simple':

            # simple - splits the data by the order they're arranged in
            # random - splits the data randomly
            m, r = divmod(self.size,k)
            Nr = range(self.size)

            if method == 'random':
                shuffle(Nr)

            parts = [ Nr[i*m:(i+1)*m + (1 if i < r else 0) ] for i in xrange(k) ]
            subsets = [ self.subset(p) for p in parts ]

        elif method == 'kd_partition':

            # splits the data using the kd tree algorithm, but stopping
            # when the desired number of partitions is achieved (must be a
            # power of 2), splitting rule: divide data into two parts
            # in the median of the largest dimension

            # make sure k is a power of 2 here, otherwise override it
            # with the closest. If k is not a power of 2, we will get the
            # largest power of 2 smaller than k.
            
            n_split = int(np.log2(k)) # int rounds down
            k = 2**n_split

            X = self.X

            Xmin, Xmax = X.min(axis=0), X.max(axis=0)

            # index of biggest dimension
            i = np.argmax(Xmax-Xmin)

            # find the mid point
            median = np.median(X[:,i])

            # split the data
            inds = [ X[:,i] > median, X[:,i] < median ]
            parts =  [ np.array(np.where(ind)).squeeze() for ind in inds ]

            if median in X[:,i]:
                meds = np.array(np.where(X[:,i] == median)).squeeze()
                nmeds = meds.size
                if nmeds == 1:
                    parts[0] = np.append(parts[0],meds)
                else:
                    parts[0] = np.append(parts[0],meds[:nmeds/2])
                    parts[1] = np.append(parts[1],meds[nmeds/2:])

            subsets = [ self.subset(p) for p in parts ]

            n_iter = 1

            while n_iter < n_split:

                subsets = [ c.partition(2,method='kd_partition') 
                            for c in subsets ]
                subsets = reduce(lambda a,b: a+b,subsets)

                n_iter += 1

        elif method == 'kd_uniform':
            # find number of partitions to obtain with the kd_partition method
            nparts = int(self.size/k)
            partitions = self.partition(nparts,'kd_partition')
            partitions_split = [ p.partition(k,method='random')
                                 for p in partitions ]

            # rearrange
            # outer list has nparts elements
            # inner lists have k DS elements
            slice = lambda i, mainlist: [ l[i] for l in mainlist ]

            subsets_list = [ slice(i,partitions_split) 
                             for i in xrange(len(partitions_split[0])) ]

            subsets = [ reduce(DataSet.join,sl) for sl in subsets_list ]

        return subsets

    def detach(self):
        # if object is a subset, detaches from superset and owns its 
        # own data
        if self.is_subset:
            self._X = self.X
            self._y = self.y
            self._superset = None
            self._indices = None

    def resolve(self):
        # if object's superset is a subset, resolves for the super-est set
        if self.superset is not None:
            while self.superset.is_subset:
                self._indices = self.superset.indices[self.indices]
                self._superset = self.superset.superset

    @property
    def cached(self):
        # cache object's data (save time on slicing superset's data)
        if self.is_subset:
            return self._X is not None

    @property
    def X(self):
        if self.is_subset:
            if self.cached:
                return self._X
            else:
                return self.superset.X[self.indices,]
        else:
            return self._X

    @property
    def y(self):
        if self.is_subset:
            if self.cached:
                return self._y
            else:
                return self.superset.y[self.indices]
        else:
            return self._y

    @property
    def superset(self):
        return self._superset

    @property
    def is_subset(self):
        return self._superset is not None

    @property
    def indices(self):
        return self._indices

    @property
    def size(self):
        return self.y.size

    @staticmethod
    def join(d1,d2):
        d = copy.copy(d1)
        d.add(d2)
        return d
