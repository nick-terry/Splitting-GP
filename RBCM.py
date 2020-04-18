# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:06:03 2020

@author: pnter
"""
import torch
import gpytorch
import itertools
import CommitteeMember
import time
'''
Implements the Robust Bayesian Committee Machine described in Deisenroth and Ng, 2015
'''

class RobustBayesCommitteeMachine():
    
    '''
    likelihoodFn: the likelihood distribution of child models. Default is Gaussian per the paper.
    
    kernel: covariance function. Default is RBF w/ no ARD. Note that in the paper ARD is used, but we need to specify the number of dims to enable.
    '''
    def __init__(self, likelihoodFn=gpytorch.likelihoods.GaussianLikelihood, kernel=gpytorch.kernels.RBFKernel, **kwargs):
            
        #Default output dimension is 1 (scalar)
        self.outputDim = 1 if 'outputDim' not in kwargs else kwargs['outputDim']
        
        self.likelihoodFn = likelihoodFn
        
        self.covar_module = kernel
            
        # of children to use
        self.numChildren = kwargs['numChildren'] if 'numChildren' in kwargs else 10    
    
        #This will contain the child models
        self.children = []
        
        #Track number of data points given to each child
        self.childrenNumData = []
        
        # Number of gradient descent iterations to use for optimization of child models
        self.training_iter = 100
        
        #We will take a N(0,1) prior, so set RBCM var to unity
        self.varStarStar = 1
        
        #Track the index of the next child to update
        self.nextChildToUpdate = 0
        
    '''
    Create a child model. This will continue with each data added, until we reach the desired # of children
    '''
    def createChild(self,x,y):
        self.children.append(CommitteeMember.CommitteeMember(self, x, y))
        
    '''
    Update the committee machine with a new (x,y) pair
    '''
    def update(self,x,y):
        #If we haven't created the desired # of children, make a new one
        if len(self.children) < self.numChildren:
            self.createChild(x, y)
            
            #Keep track of this child's data
            self.childrenNumData.append(1)
            
        else: 
            '''
            #We assume that we originally assigned the data via uniform randomness acros the children. Now choose the child with the smallest data set.
            childToUpdateIndex = self.childrenNumData.index(min(self.childrenNumData))
            '''
            childToUpdateIndex = self.nextChildToUpdate
            
            
            childToUpdate = self.children[childToUpdateIndex]
            
            t0 = time.time()
            newChildModel = childToUpdate.update(x,y)
            
            self.children[childToUpdateIndex] = newChildModel
            t1 = time.time()
            print(t1-t0)
            
            
            self.childrenNumData[childToUpdateIndex] += 1
            
            self.setChildLastUpdated(newChildModel)
            
            #Compute next child to update mod max # children
            self.nextChildToUpdate = self.nextChildToUpdate + 1 if self.nextChildToUpdate < self.numChildren-1 else 0
            
            del childToUpdate
    
    def setChildLastUpdated(self,child):
        for _child in self.children:
            _child.lastUpdated = False
        child.lastUpdated = True
    
    '''
    Predict at the points in x by soliciting (weighted) predictions from each child model
    '''
    def predict(self,x,individualPredictions=False):    
        #Update all of the covar modules to the most recent
        for child in self.children:
            child.covar_module = self.covar_module
        
        #If x is a tensor with dimension d1 x d2 x ... x dk x n, iterate over the extra dims and predict at each point
        #Create a 2D list which ranges over all values for each input dimension
        dimRangeList = [list(range(dimSize)) for dimSize in x.shape[:-1]]
        
        #Take a cross product to get all possible coordinates in the inputs
        inputDimIterator = itertools.product(*dimRangeList)
        
        #Initialize tensor of zeros to store predictions
        predictions = torch.zeros(size=(x.shape[:-1],self.outputDim))
        
        if individualPredictions:
            individualList = []
            weightsList = []
            minDistsList = []
        
        #Get individual predictions for each point
        for inputIndices in inputDimIterator:
            results = self.predictAtPoint(x[inputIndices].unsqueeze(0),individualPredictions)
            
            if individualPredictions:
                predictions[inputIndices] = results[0]
                individualList.append(results[1])
                weightsList.append(results[2])
                
            else:
                predictions[inputIndices] = results
        
        if individualPredictions:
            return predictions,individualList,weightsList
        
        else:
            return predictions
    
    '''
    Predict at a point x by soliciting a (weighted) prediction from each child model
    ''' 
    def predictAtPoint(self,x,individualPredictions=False):
        
        childPredictions = torch.zeros((len(self.children),1))
        childBetas = torch.zeros((len(self.children),1))
        childVars = torch.zeros((len(self.children),1))
        
        #Get the predictive mean and var for each child, along with betas
        for i in range(len(self.children)):
            
            child = self.children[i]
            childPredictions[i],childVars[i],childBetas[i] = child.predict(x)
            
            
        #Compute the committee's predictive var
        predVar = 1.0/(torch.sum(childBetas*childVars) - (1-torch.sum(childBetas))*self.varStarStar)
        
        #Compute the committee's predictive mean
        predMean = torch.sum(childPredictions*childVars*childBetas)*predVar
        
        if individualPredictions:
            return predMean,childPredictions,childBetas
        
        else:
            return predMean
    
            
            
        
        




        