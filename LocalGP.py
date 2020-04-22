# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:57:53 2020

@author: pnter
"""

import torch
import gpytorch

from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.memoize import add_to_cache, is_in_cache
from gpytorch.lazy.root_lazy_tensor import RootLazyTensor

import numpy as np
import itertools
import copy
from math import inf

from UtilityFunctions import updateInverseCovarWoodbury

'''
Implements the Local Gaussian Process Regression Model as described by Nguyen-tuong et al.

Note that the kernel used in the original paper Local Gaussian Process Regression for Real Time Online Model Learning uses the RBF kernel

Parameters:
    likelihoodFn: The function which, when called, instantiates a new likelihood of the type which should be used for all child models
    kernel: The kernel function used to construct the covariances matrices
    w_gen: The threshold distance for generation of a new child model

More docstring here

'''
class LocalGPModel:
    def __init__(self, likelihoodFn, kernel, inheritKernel=True, **kwargs):        
        #Initialize a list to contain local child models
        self.children = []
        self.w_gen = kwargs['w_gen'] if 'w_gen' in kwargs else .5
        self.covar_module = kernel
        self.mean_module = kwargs['mean'] if 'mean' in kwargs else gpytorch.means.ConstantMean
        self.likelihood = likelihoodFn
        self.inheritKernel = inheritKernel
        
        #Number of training iterations used each time child model is updated
        self.training_iter = 25
        
        #Default output dimension is 1 (scalar)
        self.outputDim = 1 if 'outputDim' not in kwargs else kwargs['outputDim']
        
        #If numInducingInputs is given, use variational GP models for child models
        if 'numInducingPoints' in kwargs:
            self.numInducingPoints = kwargs['numInducingPoints']
            assert(type(self.numInducingPoints)==int)
            assert(self.numInducingPoints>0)
            self.objectiveFunctionClass = gpytorch.mlls.VariationalELBO
        else:
            self.numInducingPoints = None
        
        #If maxChildren in kwargs, set self.maxChildren. Else, set to inf
        if 'maxChildren' in kwargs:
            self.maxChildren = kwargs['maxChildren']
        else:
            self.maxChildren = inf
        
        #If M=# of closest models for prediction is given, set parameter
        if 'M' in kwargs:
            self.M = kwargs['M']
        else:
            self.M = None
    '''
    Update the LocalGPModel with a pair {x,y}. x may be n-dimensional, y is scalar
    '''
    def update(self,x,y):
        #If no child model have been created yet, instantiate a new child with {x,y} and record the output dimension
        if len(self.children)==0:
            self.createChild(x,y)
            self.outputDim = int(y.shape[-1])
            
        #If child models exist, find the the child whose center is closest to x
        else:
            closestChildIndex,minDist = self.getClosestChild(x)
            
            #Check if the closest child is farther away than child model generation threshold,
            #or if the max number of children has already been generated
            if float(minDist) > self.w_gen or len(self.children) >= self.maxChildren:
                
                closestChildModel = self.children[closestChildIndex]
                #Create a new model which additionally incorporates the pair {x,y}
                newChildModel = closestChildModel.update(x,y)            
                
                #Set other child to be last updated
                self.setChildLastUpdated(newChildModel)
                
                #Replace the existing model with the new model which incorporates new data
                self.children[closestChildIndex] = newChildModel
                del closestChildModel
                
            else:
                #Since the distance from x to the nearest child model is greater than w_gen, create a new child model centered at x
                self.createChild(x,y)
    
        
    '''
    Instantiate a new child model using the training pair {x,y}
    
    Note that the likelihood used to instantiate the child model is distinct
    from each other child model, as opposed to the kernel which is shared 
    between the children.
    '''
    def createChild(self,x,y):
        #Create new child model, then train
        if self.numInducingPoints is None:
            newChildModel = LocalGPChild(x,y,self,self.inheritKernel)
        else:
            newChildModel = ApproximateGPChild(x,y,self,self.inheritKernel)
        
        #Set other children to not be last updated.
        self.setChildLastUpdated(newChildModel)
        
        #Add to the list of child models
        self.children.append(newChildModel)
    
    def setChildLastUpdated(self,child):
        for _child in self.children:
            _child.lastUpdated = False
        child.lastUpdated = True
            
    
    '''
    Return a pytorch tensor of the centers of all child models.
    '''
    def getCenters(self):
        #Get the center of each child model
        centersList = list(map(lambda x:x.center.reshape((x.center.shape[0])),self.children))
        #Return the centers after stacking in new dimension
        return torch.stack(centersList,dim=0)
    
    '''
    Returns the index of the closest child model to the point x, as well as the distance
    between the model's center and x.
    '''
    def getClosestChild(self,x):
        #Compute distances between new input x and existing inputs
        distances = self.getDistanceToCenters(x)
        #Get the single minimum distance from the tensor (max covar)
        minResults = torch.max(distances,1)
        return minResults[1],minResults[0]
    
    '''
    Compute the distances from the point x to each center
    '''
    def getDistanceToCenters(self,x,returnPowers=False):
        centers = self.getCenters()
        x,centers = x.double(),centers.double()
        distances = self.covar_module(x,centers).evaluate()
        powers = torch.zeros(distances.shape)
        #Switch to double precision for this calculation
        '''
        vec = ((x-centers)/self.covar_module.lengthscale).double()
        powers = .5*torch.sum(vec**2,dim=1)
        distances = torch.exp(-powers)
        '''
        
        if returnPowers:
            return distances.squeeze(0),powers
        else:
            return distances.squeeze(0)
    
    '''
    Make a prediction at the point(s) x. This method is a wrapper which handles the messy case of multidimensional inputs.
    The actual prediction is done in the predictAtPoint helper method. If no M is given, use default
    '''
    def predict(self,x,individualPredictions=False):
        return self.predict(x,self.M,individualPredictions)
    
    '''
    Make a prediction at the point(s) x. This method is a wrapper which handles the messy case of multidimensional inputs.
    The actual prediction is done in the predictAtPoint helper method
    '''
    def predict(self,x,M=None,individualPredictions=True):
        
        #Update all of the covar modules to the most recent
        for child in self.children:
            child.covar_module = self.covar_module
            
        #If x is a tensor with dimension d1 x d2 x ... x dk x n, iterate over the extra dims and predict at each point
        #Create a 2D list which ranges over all values for each input dimension
        dimRangeList = [list(range(dimSize)) for dimSize in x.shape[:-1]]
        
        #Take a cross product to get all possible coordinates in the inputs
        inputDimIterator = itertools.product(*dimRangeList)
        
        #Initialize tensor of zeros to store predictions
        predictions = torch.zeros(size=(*x.shape[:-1],self.outputDim))
        
        if individualPredictions:
            individualList = []
            weightsList = []
            minDistsList = []
        
        for inputIndices in inputDimIterator:
            results = self.predictAtPoint(x[inputIndices].unsqueeze(0),M,individualPredictions)
            
            if individualPredictions:
                predictions[inputIndices] = results[0]
                individualList.append(results[1])
                weightsList.append(results[2])
                minDistsList.append(results[3])
            else:
                predictions[inputIndices] = results
        
        if individualPredictions:
            return predictions,individualList,weightsList,minDistsList
        
        else:
            return predictions
    '''
    Make a prediction at the point x by finding the M closest child models and
    computing a weighted average of their predictions. By default M is the number
    of child models. If M < number of child models, use all of them.
    '''
    def predictAtPoint(self,x,M=None,individualPredictions=False):
        if M is None:
            M = len(self.children)
        else:
            M = min(M,len(self.children))
        
        
        #Compute distances between new input x and existing inputs
        distances,powers = self.getDistanceToCenters(x,True)
        
        #Get the M closest child models. Need to squeeze out extra dims of 1.
        sortResults = torch.sort(distances.squeeze(-1).squeeze(-1),descending=True)
        minDists = sortResults[0][:M].squeeze(-1) if sortResults[0].dim()>0 else sortResults[0].unsqueeze(0)
        minIndices = sortResults[1][:M] if sortResults[1].dim()>0 else sortResults[1].unsqueeze(0)
        closestChildren = [self.children[i] for i in minIndices]
        
        '''
        Get a posterior distribution for each child model. Note each will be
        multivariate normal. Then compute weighted average of the means of the
        posterior distributions.
        '''
        posteriorMeans = []
        for child in closestChildren:
            posterior = child.predict(x)
            posteriorMeans.append(posterior.mean)
        
        '''
        TODO: It would be better to instead compute the weighted average of the
        posterior distributions so we have access to variance as well. Presumably
        the resulting distribution is also multivariate normal.
        '''
        
        
        posteriorMeans = torch.stack(posteriorMeans)
        
        #We need to be careful with this computation. If the covariances are very small, we may end up with a nan value here.
        nonZeroDists = minDists[minDists>0.0]
        #Address the case where we are predicting very far away from all models. Take unweighted mean of all predictions
        if nonZeroDists.shape[-1]==0:
            weights = 1.0/(powers+1.0)
            weights = weights/torch.sum(weights)
        else:
            minDists = minDists
            weights = minDists/torch.sum(minDists)
        weightedAverageMean = torch.dot(weights,posteriorMeans.squeeze(-1).double()).float()
        
        '''
        negLogDists = -torch.log(minDists)
        weights = 1.0/(negLogDists/torch.sum(negLogDists))
        weights = weights/torch.sum(weights)
        weightedAverageMean = torch.dot(weights,posteriorMeans.squeeze(-1))
        '''
        
        if individualPredictions:
            return weightedAverageMean,posteriorMeans,weights,minDists

        else:
            return weightedAverageMean
        
class LocalGPChild(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, parent, inheritKernel=True, **kwargs):
        
        #Track if the child was created by splitting
        self.isSplittingChild = True if 'split' in kwargs and kwargs['split'] else False
        
        #Handle prior likelihood
        if 'priorLik' in kwargs and kwargs['priorLik'] is not None:
            priorLik =  kwargs['priorLik']
        else:
            #If no prior is provided, use the default of the parent
            priorLik = parent.likelihood()
            
            #In this case, we reset the isSplittingChild flag to false in order for the new likelihood to be trained
            self.isSplittingChild = False
            
        super(LocalGPChild, self).__init__(train_x, train_y, priorLik)
        
        self.parent = parent
        
        if 'priorMean' in kwargs and kwargs['priorMean'] is not None:
            #If given, take a prior for the mean. Used for splitting models.
            self.mean_module = copy.deepcopy(kwargs['priorMean'])
        else:
            self.mean_module = parent.mean_module()
        
        '''
        If inheritKernel is set to True, then the same Kernel function (including the same hyperparameters)
        will be used in all of the child models. Otherwise, a separate instance of the same kernel function
        is used for each child model.
        '''
        if inheritKernel:
            self.covar_module = parent.covar_module
        else:
            self.covar_module = copy.deepcopy(parent.covar_module)
        self.lastUpdated = True
        
        '''
        Compute the center as the mean of the training data
        '''
        self.center = torch.mean(train_x,dim=0)
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
    '''
    def update(self,x,y):
        if self.prediction_strategy is None:
            self.predict(x)
        
        
        #If this was the last child to be updated, or inheritKernel=False, we can do a fantasy update
        #without updating the covar cache. Otherwise, we can update the covar_module from the parent
        
        if not self.lastUpdated and self.parent.inheritKernel:
                self.covar_module = self.parent.covar_module
                self.predict(x)
        
        #Sometimes get an error when attempting Cholesky decomposition.
        #In this case, refit a new model.
        
        try:
            
            updatedModel = self.get_fantasy_model(inputs=x, targets=y)
        
        except RuntimeError as e:
            print('Error during Cholesky decomp for fantasy update. Fitting new model...')
            
            newInputs = torch.cat([self.train_x,x],dim=0)
            newTargets = torch.cat([self.train_y,y],dim=0)
            updatedModel = LocalGPChild(newInputs,newTargets,self.parent,
                                        inheritKernel=self.parent.inheritKernel)
        
        except RuntimeWarning as e:
            print('Error during Cholesky decomp for fantasy update. Fitting new model...')
            
            newInputs = torch.cat([self.train_x,x],dim=0)
            newTargets = torch.cat([self.train_y,y],dim=0)
            updatedModel = LocalGPChild(newInputs,newTargets,self.parent,
                                        inheritKernel=self.parent.inheritKernel)
        
        #Update the data properties
        updatedModel.train_x = updatedModel.train_inputs[0]
        updatedModel.train_y = updatedModel.train_targets
        
        #Compute the center of the new model
        updatedModel.center = torch.mean(updatedModel.train_inputs[0],dim=0)
        
        #Update parent's covar_module
        self.parent.covar_module = self.covar_module
        
        #Need to perform a prediction so that get_fantasy_model may be used to update later
        updatedModel.predict(x)
        
        return updatedModel
    '''
    def update(self,x,y):
        #Sync covar
        self.covar_module = self.parent.covar_module
        
        #Update train_x, train_y
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, y])
        
        #Update the data which can be used for optimizing
        self.train_inputs = (self.train_x,)
        self.train_targets = self.train_y
        
        self.retrain()
        
        return self
    
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
                #Compute the update
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
        
        #We only train on instantiation if the child model is not a result of a split
        if not self.isSplittingChild:
            #Setup optimizer
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
            
            #Perform training iterations
            for i in range(self.parent.training_iter):
                self.optimizer.zero_grad()
                output = self(self.train_x)
                loss = -mll(output, self.train_y)
                try:
                    loss.backward()
                except:
                    print(loss)
                    print(self.train_y)

                    
                self.optimizer.step()
            
        #Need to perform a prediction so that get_fantasy_model may be used to update later
        self.predict(self.train_x)
    
    '''
    Retrain model after new data is obtained
    '''
    def retrain(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        #Perform training iterations
        for i in range(self.parent.training_iter):
            self.optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if loss < 0:
                break
            else:
                self.optimizer.step()
    
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
        
'''
A child model which computes an approximate posterior distribution using inducing points.
'''
class ApproximateGPChild(gpytorch.models.VariationalGP):
    def __init__(self, train_x, train_y, parent, inheritKernel=True):
        #Set the number of inducing inputs to be no more than the number of points observed
        numInducingPoints = min(parent.numInducingPoints,train_x.shape[0])
        #Define the multivariate normal variational distribution for approximate posterior computation
        variationalDist = gpytorch.variational.CholeskyVariationalDistribution(numInducingPoints)
        variationalStrat = gpytorch.variational.VariationalStrategy(self,train_x,variationalDist,learn_inducing_locations=True)
        
        super(ApproximateGPChild, self).__init__(variationalStrat)
        
        self.parent = parent
        self.mean_module = gpytorch.means.ConstantMean()
        self.variationalDist = variationalDist
        self.variationalStrat = variationalStrat
        self.numInducingPoints = numInducingPoints
        
        self.likelihood = parent.likelihood()
        '''
        If inheritKernel is set to True, then the same Kernel function (including the same hyperparameters)
        will be used in all of the child models. Otherwise, a separate instance of the same kernel function
        is used for each child model.
        '''
        if inheritKernel:
            self.covar_module = parent.covar_module
        else:
            self.covar_module = copy.deepcopy(parent.covar_module)
        '''
        Since the data is assumed to arrive as pairs {x,y}, we assume that the 
        initial train_x,train_y are singleton, and set train_x as the intial
        center for the model.
        '''
        self.center = train_x
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
        #Concatenate existing data with new data
        self.train_x = torch.cat([self.train_x,x],dim=0)
        self.train_y = torch.cat([self.train_y,y],dim=0)
        
        #Compute the center of the model
        self.center = torch.mean(self.train_x,dim=0)
        
        #Set the number of inducing inputs to be no more than the number of points observed
        self.numInducingPoints = min(self.parent.numInducingPoints,self.train_x.shape[0])
        
        #Create new variational strategy for the updated model
        self.variationalStrat = gpytorch.variational.VariationalStrategy(
                model=self,
                inducing_points=self.train_x,
                variational_distribution=self.variationalDist,
                learn_inducing_locations=True)
        
        #Retrain to fit new data
        self.retrain()
        return self
    
    '''
    Setup optimizer and perform initial training
    '''
    def initTraining(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        #Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        objectiveFunction = self.parent.objectiveFunctionClass(self.likelihood,self,self.train_x.shape[0])
        
        #Perform training iterations
        for i in range(self.parent.training_iter):
            self.optimizer.zero_grad()
            output = self(self.train_x)
            loss = -objectiveFunction(output, self.train_y)
            loss.backward()
            self.optimizer.step()
    
    
    
    '''
    Retrain model after new data is obtained
    '''
    def retrain(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        objectiveFunction = self.parent.objectiveFunctionClass(self.likelihood,self,self.train_x.shape[0])
        
        #Perform training iterations
        for i in range(self.parent.training_iter):
            self.optimizer.zero_grad()
            output = self(self.train_x)
            loss = -objectiveFunction(output, self.train_y)
            loss.backward()
            self.optimizer.step()
    
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