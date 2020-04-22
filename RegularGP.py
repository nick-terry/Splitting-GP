# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 07:45:02 2020

@author: pnter
"""

import torch
import gpytorch

from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.memoize import add_to_cache, is_in_cache

from varBoundFunctions import *
import numpy as np
import itertools
import copy
from math import inf

from UtilityFunctions import updateInverseCovarWoodbury

'''
Implements a generic Exact GP model, but with the same interface as our LocalGP model.
To be used for comparison experiments

Parameters:
    likelihoodFn: The function which, when called, instantiates a new likelihood of the type which should be used for all child models
    kernel: The kernel function used to construct the covariances matrices

More docstring here

'''
class RegularGPModel:
    def __init__(self, likelihoodFn, kernel, inheritKernel=True, fantasyUpdate=True, **kwargs):        
        #Initialize a list to contain local child models
        self.covar_module = kernel
        self.mean_module = kwargs['mean'] if 'mean' in kwargs else gpytorch.means.ConstantMean
        self.likelihood = likelihoodFn
        
        self.children = []
        
        self.fantasyUpdate = fantasyUpdate
        self.inheritKernel = inheritKernel
        
        #Number of training iterations used each time child model is updated
        self.training_iter = 500
        
        #Default output dimension is 1 (scalar)
        self.outputDim = 1 if 'outputDim' not in kwargs else kwargs['outputDim']
        
        self.child = None
        
    '''
    Update the LocalGPModel with a pair {x,y}. x may be n-dimensional, y is scalar
    '''
    def update(self,x,y):
        #If no child model have been created yet, instantiate a new child with {x,y} and record the output dimension
        if self.child is None:
            self.createChild(x,y)
            self.outputDim = int(y.shape[-1])
            
        #If child models exist, find the the child whose center is closest to x
        else:
            self.child = self.child.update(x,y)
    
    '''
    Instantiate a new child model using the training pair {x,y}
    '''
    def createChild(self,x,y):
        newChildModel = RegularGPChild(x,y,self,self.inheritKernel,self.fantasyUpdate)
        
        self.child = newChildModel            
    
    '''
    Returns the index of the closest child model to the point x, as well as the distance
    between the model's center and x.
    '''
    def getClosestChild(self,x):
        #Compute distances between new input x and existing inputs
        distances = self.getDistanceToCenters(x)
        #Get the single minimum distance from the tensor
        minResults = torch.max(distances,0)
        return minResults[1],minResults[0]
    
    '''
    Compute the distances from the point x to each center
    '''
    def getDistanceToCenters(self,x):
        centers = self.getCenters()
        distances = self.covar_module(x,centers).evaluate()
        return distances.squeeze(0)
    
    '''
    Make a prediction at the point(s) x. This method is a wrapper which handles the messy case of multidimensional inputs.
    The actual prediction is done in the predictAtPoint helper method. If no M is given, use default
    '''
    def predict(self,x):
        return self.predict(x,self.M)
    
    '''
    Make a prediction at the point(s) x. This method is a wrapper which handles the messy case of multidimensional inputs.
    The actual prediction is done in the predictAtPoint helper method
    '''
    def predict(self,x,M=None):
        #If x is a tensor with dimension d1 x d2 x ... x dk x n, iterate over the extra dims and predict at each point
        #Create a 2D list which ranges over all values for each input dimension
        dimRangeList = [list(range(dimSize)) for dimSize in x.shape[:-1]]
        
        #Take a cross product to get all possible coordinates in the inputs
        inputDimIterator = itertools.product(*dimRangeList)
        
        #Initialize tensor of zeros to store predictions
        predictions = torch.zeros(size=(*x.shape[:-1],self.outputDim))
        
        for inputIndices in inputDimIterator:
            predictions[inputIndices] = self.predictAtPoint(x[inputIndices].unsqueeze(0),M)
        
        return predictions
    
    '''
    Make a prediction at the point x by finding the M closest child models and
    computing a weighted average of their predictions. By default M is the number
    of child models. If M < number of child models, use all of them.
    '''
    def predictAtPoint(self,x,M=None):
        posterior = self.child.predict(x)
        
        return posterior.mean
        
class RegularGPChild(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, parent, inheritKernel=True, fantasyUpdate=False):
        super(RegularGPChild, self).__init__(train_x, train_y, parent.likelihood())
        
        self.parent = parent
        self.mean_module = parent.mean_module()
        
        self.covar_module = parent.covar_module
        
        self.fantasyUpdate = fantasyUpdate
        
        self.train_x = train_x
        self.train_y = train_y
        
        self.initTraining()
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    '''
    Update the child model to incorporate the training pair {x,y}
    '''
    def update(self,x,y):
        if self.fantasyUpdate:
            if self.prediction_strategy is None:
                self.predict(x)
            
            '''
            Sometimes get an error when attempting Cholesky decomposition.
            In this case, refit a new model.
            '''
            try:
                
                updatedModel = self.get_fantasy_model(inputs=x, targets=y)
            
            except RuntimeError as e:
                print('Error during Cholesky decomp for fantasy update. Fitting new model...')
                
                newInputs = torch.cat([self.train_x,x],dim=0)
                newTargets = torch.cat([self.train_y,y],dim=0)
                updatedModel = RegularGPChild(newInputs,newTargets,
                                          self.parent,
                                          inheritKernel=self.parent.inheritKernel,
                                          fantasyUpdate=self.fantasyUpdate)
            
            except RuntimeWarning as e:
                print('Error during Cholesky decomp for fantasy update. Fitting new model...')
                
                newInputs = torch.cat([self.train_x,x],dim=0)
                newTargets = torch.cat([self.train_y,y],dim=0)
                updatedModel = RegularGPChild(newInputs,newTargets,
                                          self.parent,
                                          inheritKernel=self.parent.inheritKernel,
                                          fantasyUpdate=self.fantasyUpdate)
            
            #Update the data properties
            updatedModel.train_x = updatedModel.train_inputs[0]
            updatedModel.train_y = updatedModel.train_targets
            
            #Need to perform a prediction so that get_fantasy_model may be used to update later
            updatedModel.predict(x)
        
        else:
            newInputs = torch.cat([self.train_x,x],dim=0)
            newTargets = torch.cat([self.train_y,y],dim=0)
            
            updatedModel = RegularGPChild(newInputs,newTargets,
                                          self.parent,
                                          inheritKernel=self.parent.inheritKernel,
                                          fantasyUpdate=self.fantasyUpdate)
        
        return updatedModel
    
    '''
    Perform a rank-one update of the child model's inverse covariance matrix cache.
    This is necessary if the model is is NOT the most recently updated child model.
    '''
    def updateInvCovarCache(self,update=False):
        lazy_covar = self.prediction_strategy.lik_train_train_covar
        if is_in_cache(lazy_covar,"root_inv_decomposition"):
            if update:
                #Get the old cached inverse covar matrix 
                K_0inv = lazy_covar.root_inv_decomposition()
                #Get the new covar matrix by calling the covar module on the training data
                K = self.covar_module(self.train_x)
                #Compute the rank-one update
                Kinv = updateInverseCovarWoodbury(K_0inv, K)
                #Store updated inverse covar matrix in cache
                add_to_cache(lazy_covar, "root_inv_decomposition", RootLazyTensor(torch.sqrt(Kinv)))
            else:
                #This is a bit dirty, but here we will simply delete the root/root_inv from cache. This forces
                #GPyTorch to recompute them.
                
                lazy_covar._memoize_cache = {}
                self.prediction_strategy._memoize_cache = {}
                
    '''
    Setup optimizer and perform initial training
    '''
    def initTraining(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        #Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        #Perform training iterations
        for i in range(self.parent.training_iter):
            self.optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            self.optimizer.step()
        
        #Need to perform a prediction so that get_fantasy_model may be used to update later
        self.predict(self.train_x)
    '''
    Evaluate the child model to get the predictive posterior distribution
    '''
    def predict(self,x):
        #Switch to eval/prediction mode
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self(x))
        
        return prediction