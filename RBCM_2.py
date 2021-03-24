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
    Update the committee machine with a new (x,y)
    '''
    def update(self,x,y):
        if y.dim()==1:
            y = y.unsqueeze(-1)
        #If we haven't created the desired # of children, make a new one
        if len(self.children) < self.numChildren:
            self.createChild(x[0,:].unsqueeze(0), y[0,:])
            
            #Keep track of this child's data
            self.childrenNumData.append(1)
            
            #Recursively update until we have enough children
            if x.shape[0] > 1:
                self.update(x[1:,:],y[1:,:])
            
            return
            
        else:
            #Generate a U(0,1) random vector, then scale to the number of children
            assignments = torch.floor(torch.rand((x.shape[0],1))*self.numChildren)
            
            for i in range(self.numChildren):
                child = self.children[i]
                
                assignToChild = (assignments==i)
                
                if torch.sum(assignToChild) > 0:
                    assignToChild = assignToChild.squeeze()
                    x_assign = x[assignToChild]
                    y_assign = y[assignToChild]
                    
                    y_assign = y_assign if y_assign.dim()>1 else y_assign.unsqueeze(0)
                
                    child.update(x_assign,y_assign)
              
            '''
            #We assume that we originally assigned the data via uniform randomness across
            the children. Now choose the child with the smallest data set.
            '''
            '''
            childToUpdateIndex = self.nextChildToUpdate
            
            
            childToUpdate = self.children[childToUpdateIndex]
            
            childToUpdate.update(x,y)
            
            self.childrenNumData[childToUpdateIndex] += 1
            
            self.setChildLastUpdated(childToUpdate)
            
            #Compute next child to update mod max # children
            self.nextChildToUpdate = self.nextChildToUpdate + 1 if self.nextChildToUpdate < self.numChildren-1 else 0
            '''
    
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
        
        mean_predictions = []
        var_predictions = []
        betas_predictions = []
        
        #Get the predictions of each child at each point
        for child in self.children:
            prediction,betas = child.predict(x)
            
            mean_predictions.append(prediction.mean)
            var_predictions.append(prediction.variance)
            betas_predictions.append(betas)
            
        #Concatenate into pytorch tensors
        mean_predictions = torch.stack(mean_predictions).transpose(0,1)
        var_predictions = torch.stack(var_predictions).transpose(0,1)
        betas = torch.stack(betas_predictions).transpose(0,1)
        
        #Compute the committee's predictive var
        predVar = (1.0/(torch.sum(betas/var_predictions,dim=1) + (1-torch.sum(betas,dim=1))/self.varStarStar)).unsqueeze(-1)
        
        #Compute the committee's predictive mean
        predMean = torch.sum(mean_predictions/var_predictions*betas,dim=1).unsqueeze(-1)*predVar
        
        return predMean,predVar
        
    '''
        ########################################################
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
            childResults = child.predict(x)
            #Switch to double precision for this critical computation
            childPredictions[i],childVars[i],childBetas[i] = [results.double() for results in childResults]
            
            
        #Compute the committee's predictive var
        predVar = 1.0/(torch.sum(childBetas/childVars) + (1-torch.sum(childBetas))/self.varStarStar)
        
        #Compute the committee's predictive mean
        predMean = torch.sum(childPredictions/childVars*childBetas)*predVar
        
        if individualPredictions:
            return predMean,childPredictions,childBetas
        
        else:
            return predMean
    
            
            
        
        




        