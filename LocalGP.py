# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:57:53 2020

@author: pnter
"""

import torch
import gpytorch
from varBoundFunctions import *
import numpy as np
import itertools
import copy

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
    def __init__(self, likelihoodFn, kernel, w_gen, inheritKernel=True):        
        #Initialize a list to contain local child models
        self.children = []
        self.w_gen = w_gen
        self.covar_module = kernel
        self.likelihood = likelihoodFn
        self.inheritKernel = inheritKernel
        
        #Number of training iterations used each time child model is updated
        self.training_iter = 200
        
        #Default output dimension is 1 (scalar)
        self.outputDim = 1
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
            
            #Check if the closest child is farther away than child model generation threshold
            if float(minDist) > self.w_gen:
                
                closestChildModel = self.children[closestChildIndex]
                #Create a new model which additionally incorporates the pair {x,y}
                newChildModel = closestChildModel.update(x,y)
                #Replace the existing model with the new model which incorporates new data
                self.children[closestChildIndex] = newChildModel
                
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
        newChildModel = LocalGPChild(x,y,self,self.inheritKernel)
    
        #Add to the list of child models
        self.children.append(newChildModel)
        
    '''
    Return a pytorch tensor of the centers of all child models.
    '''
    def getCenters(self):
        #Get the center of each child model
        centersList = list(map(lambda x:x.center.reshape((2)),self.children))
        #Return the centers after stacking in new dimension
        return torch.stack(centersList,dim=0)
    
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
        if M is None:
            M = len(self.children)
        else:
            M = min(M,len(self.children))
        
        
        #Compute distances between new input x and existing inputs
        distances = self.getDistanceToCenters(x)
        
        #Get the M closest child models. Need to squeeze out extra dims of 1.
        sortResults = torch.sort(distances.squeeze(-1).squeeze(-1),descending=True)
        minDists,minIndices = sortResults[0][:M].squeeze(-1),sortResults[1][:M]
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
        #This computation is incorrect
        weightedAverageMean = torch.dot(minDists,posteriorMeans.squeeze(-1))/torch.sum(minDists)
        
        return weightedAverageMean
        
class LocalGPChild(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, parent, inheritKernel=True):
        super(LocalGPChild, self).__init__(train_x, train_y, parent.likelihood())
        
        self.parent = parent
        self.mean_module = gpytorch.means.ConstantMean()
        
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
        
        #These properties are for computing the posterior variance bounds
        self.k = None
        self.kinv = None
        self.kinvInnerSums = None
        
        self.initTraining()
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    '''
    Update the child model to incorporate the training pair {x,y}
    '''
    def update(self,x,y):
        updatedModel = self.get_fantasy_model(inputs=x, targets=y)
        #Compute the center of the new model
        updatedModel.center = torch.mean(updatedModel.train_inputs[0],dim=0)
        
        #Need to perform a prediction so that get_fantasy_model may be used to update later
        updatedModel.predict(x)
        return updatedModel
    
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
        
    '''
    Compute the posterior variance bounds for an **isotropic** kernel
    
    "direct" method computes posterior variance bounds directly using the 
    inverse kernel matrix and the assumption that distance between consecutive
    training data is at most tau apart
    
    "eigen" method computes the bounds using the maximal eigenvalue of the kernel matrix
    '''
    def postVarBound(self,x,train_x,**kwargs):
        #If the kernel matrix is not yet computed, compute it
        n = train_x.shape[0]
        if self.k is None:
            #Result should be square nxn tensor
            train_x = train_x.view([n,1])
            self.k = self.covar_module(train_x,train_x).evaluate()
        
        if kwargs['method'] is None:
            kwargs['method'] = 'direct'
        
        #Compute raw bounds for posterior variance directly by inverting the kernel matrix
        if kwargs['method'] is 'direct':
            if self.kinv is None:
                self.kinv = torch.inverse(self.k)
                
            #Compute the inner summations needed for the posterior variance bounds.
            #Result will be a jx1 tensor where result[j] is the sum of elements in 
            #jth column of kinv
            def innerSums():
                return torch.sum(self.kinv,dim=0)
             
            if self.kinvInnerSums is None:
                self.kinvInnerSums = innerSums()
            
            #Define (n-1)x1 tensor of values at which to evaluate the kernel
            
            boundVals = torch.tensor([((n-j+1)*tau)**2 for j in range(1,n)]).view([n-1,1])
            
            #Compute kernel evaluated for |x-x_n|, then add to kernel evaluated at boundVals and multiply by corresponding column sum from kinv
            #k_x_xn = self.covar_module(x.view([1,1]),train_x[-1].view([1,1]))**2
            k_x_xn = self.covar_module(x.view([1,1]),train_x[-1].view([1,1])).evaluate()**2
            outerSumTerms = self.covar_module(boundVals,torch.zeros(n-1).view([n-1,1]),diag=True)**2*self.kinvInnerSums[:-1]
            #Compute kernel evaluated at d=0, then return bounds
            zero_tensor = torch.zeros([1,1])
            k_0 = self.covar_module(zero_tensor,zero_tensor).evaluate()
            return k_0 - torch.sum(outerSumTerms) - k_x_xn * self.kinvInnerSums[-1]
        
        if kwargs['method'] is 'eigen':
            #Compute kernel evaluated at d=0
            zero_tensor = torch.zeros([1,1])
            k_0 = self.covar_module(zero_tensor,zero_tensor).evaluate()
            
            #Find eigenvalues of k. We can safely take the real part since
            #k is positive definite and therefore has only positive real eigenvalues.
            eigVals = np.real(np.linalg.eigvals(self.k.detach()))
            
            #Return bound
            return float(k_0 - (torch.norm(self.covar_module(x+torch.zeros(n),train_x,diag=True))**2)/np.max(eigVals))
    