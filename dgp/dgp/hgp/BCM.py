# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:29:15 2015

@author: marc
"""

import pdb
from random import shuffle
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy import linalg, optimize

import cov
from GP import GP, GP_NLML_wrapper, GP_predict_wrapper
from DataSet import DataSet
from hgp import HGP

class BCM(GP):

    def __init__(self,X,y,cov_type='covSEard',
                 profile=[(4,'kd_partition','rep2')]*2,
                 pool=None):

        # architecture of the HGP is specified by a list of tuples
        # the i-th tuple in the list represents the specification of a HGP
        # node at the i-th level (where level 0 is the root/top)
        # each tuple has the form (c,pmethod,cmethod)
        # where: c denotes the number of child GPs
        #        pmethod denotes the method used to partition the training data
        #        cmethod denotes the method used to combine the partitions

        # the argument pool can be a Pool object from the multiprocessing
        # library. If 'default', a Pool will be created, otherwise, the HGP
        # will be utilise single threaded computation
        if pool == 'default':
            self.pool = Pool(cpu_count())
        else:
            self.pool = pool

        self.X = X # training inputs, 2d array
        self.y = y # training outputs, 1d array
        self.cov_type = cov_type # string specifying covariance function

        # initialise all log params to 0
        if self.cov_type == 'covSEiso':
            self.params = np.asarray([0.0]*3)
            self.params[-1] = np.log(0.01)
        elif self.cov_type == 'covSEard':
            self.params = np.asarray([0.0]*(self.input_dim+2))
            self.params[-1] = np.log(0.01)
        
        # build the HGP
        self.build(profile)

    def build(self,profile):
        
        # construct the HGP based on the given profile

        # construction rule at the current level
        current_level = profile[0]
        c, pmethod, cmethod = current_level

        # construction rule for the next levels
        next_levels = profile[1:]

        # set up a DataSet object
        data = DataSet(self.X,self.y)

        # split data according to the partitioning method
        partitions = data.partition(c, pmethod)

        # recombine data
        if cmethod.startswith('rep'):
            # rep<n> = number of partitions per subset
            k = int(cmethod[3:])

            assert k < c # k == c implies duplication of the full GP c times

            # shuffle the partitions
            shuffle(partitions)

            # partition assignment
            # child GP  |  
            #        1  | 1,2,...k
            #        2  | 2,3,...k+1
            #        3  | 3,4,...k+2

            self.children = []

            for i in xrange(c): 
                l, u = i,i+k
                if u < c:
                    selected_parts = partitions[l:u]
                elif u >= c:    
                    selected_parts = partitions[l:] + partitions[:u-c]

                subset = reduce(DataSet.join,selected_parts)

                #print l,u, len(selected_parts), subset.size
                #pdb.set_trace()

                if len(next_levels) == 0:
                    child = GP(subset.X,subset.y,cov_type=self.cov_type)
                else:
                    child = HGP(subset.X,subset.y,cov_type=self.cov_type,
                                profile=next_levels,pool=self.pool)

                self.children.append(child)

    def NLML(self,params=None,derivs=False):

        # computes the NLML, = mean of children NLML
        
        params = self.params if params is None else params

        if self.pool == None or self.height > 1:
            NLMLs = [ c.NLML(params,derivs) for c in self.children ]
        else:
            arglist = zip(self.children,
                          [params]*self.nchild,
                          [derivs]*self.nchild)
            NLMLs = self.pool.map_async(GP_NLML_wrapper,arglist).get()

        return reduce(lambda a,b: a+b, NLMLs)/float(self.nchild)

    def predict(self,Xp,variance=False,latent_variance=False,entropy=False):
        for c in self.children:
                c.params = self.params
        if self.pool == None or self.height > 1:
            args = Xp, False, True, True 
            predictions = [ c.predict(*args) for c in self.children ]
        else:
            # predictions at the leaves (GP experts)
            arglist = zip(self.children,
                          [Xp]*self.nchild,
                          [False]*self.nchild,
                          [True]*self.nchild,
                          [True]*self.nchild)
            predictions = self.pool.map_async(GP_predict_wrapper,arglist).get()

        # child-GPs mean predictions, latent variance and entropy
        predictions_ymu = [ p[0] for p in predictions ]
        predictions_fs2 = [ p[1] for p in predictions ]
       # predictions_ent = [ p[2] for p in predictions ]

        preds = np.vstack(predictions_ymu).T

#        # compute weights
#        weights = [ e/p for p,e in zip(predictions_fs2,predictions_ent) ]
#        weights = np.vstack(weights).T
#
#        # normalise weights
#        weights_norm = weights/weights.sum(axis=1).reshape(-1,1)
#        weights_norm = np.asarray(np.nan_to_num(weights_norm))
#        
#        # compute final prediction
#        ymu = np.sum(preds*weights_norm,axis=1)


        ########### Marc's edits - compute final predictions at top
        S2 = np.vstack(predictions_fs2).T     
        precBCM = np.sum(1/S2,axis=1)
        precBCM += (1-self.nleaf)*np.exp(-self.params[-2])    
        SBCM = 1/precBCM
        ymu =SBCM*np.sum(preds/S2,axis=1)
        ###########

        
        output = (ymu,)

        if variance or latent_variance:

            # fs2 = 1/np.sum(weights,axis=1)
            
           ########### Marc's edits 
            fs2 = SBCM


           ###########
            if variance:
                ys2 = fs2 + np.exp(self.params[-1])
                output += (ys2,)

            if latent_variance:
                output +=  (fs2,)

            if entropy:
                # entropy needed if this HGP is a child-GP of another HGP
                prior_variance = np.exp(self.params[-2])
#                prior_variance = cov.cov(self.cov_type,Xp,Xp, 
#                                         self.params[:-1],diag=True)
                # prior_variance = np.asarray(prior_variance).squeeze()
                tcov = prior_variance - fs2
                output += (tcov,)

        return output[0] if len(output) == 1 else output

    @property
    def nchild(self):
        return len(self.children)

    @property
    def nleaf(self):
        if type(self.children[0]) is GP:
            return self.nchild
        else:
            return reduce(lambda a,b: a+b,[ c.nleaf for c in self.children ])

    @property
    def height(self):
        if type(self.children[0]) is GP:
            return 1
        else:
            return self.children[0].height + 1
