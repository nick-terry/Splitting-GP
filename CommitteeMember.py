# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:35:04 2020

@author: pnter
"""
import torch
import gpytorch

'''
Implements the child models described in Distributed Gaussian Processes by Deisenroth and Ng, 2015
'''
class CommitteeMember(gpytorch.models.ExactGP):
    
    def __init__(self, parent, train_x, train_y):
        
        likelihood = parent.likelihoodFn()
        
        super(CommitteeMember, self).__init__(train_x, train_y, likelihood)
        
        self.parent = parent
        
        #These are inherited from the parent
        self.likelihood = likelihood
        
        #Save the prior variance for computation of differential entropy
        self.varStarStar = self.parent.varStarStar
        
        self.covar_module = self.parent.covar_module
        
        #We assume a constant mean since it is not explicitly stated in the paper, and this is what the other models are using
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.train_x = train_x
        self.train_y = train_y
        
        self.initTraining()
        
        #Track which child was last updated
        self.lastUpdated = True
        
    #Get a prediction. Returns predictive mean, variance, and beta factor.
    #To save some time, only compute beta etc if needed
    def predict(self,x,needPred=True):
        if not self.trained:
            self.initTraining()
        #Switch to eval/prediction mode
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posteriorDist = self.likelihood(self(x))
        
        if needPred:
            #Take mean and var of dist
            mean,var = posteriorDist.mean,posteriorDist.variance
        
            beta = -.5*torch.log(var)
        
            return mean,var,beta
    
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
        self.trained = True
        self.predict(self.train_x[0,:].unsqueeze(0),False)
        
        
        
    '''
    Defines the forward pass used to optimize the model
    '''
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    '''
    Do a simple update of the training data with, training occuring only when explicitly called
    '''
    def update(self,x,y):
        self.train_x = torch.cat([self.train_x,x],dim=0)
        self.train_y = torch.cat([self.train_y,y],dim=0)
        
        self.train_inputs = (self.train_x,)
        self.train_targets = self.train_y
        self.trained = False
    
    '''
    Update the child model to incorporate the training pair {x,y}
    '''
    def fantUpdate(self,x,y):
        if self.prediction_strategy is None:
            self.predict(x,False)
        
        '''
        If this was the last child to be updated, or inheritKernel=False, we can do a fantasy update
        without updating the covar cache. Otherwise, we can update the covar_module from the parent
        '''
        
        if not self.lastUpdated:
                self.covar_module = self.parent.covar_module
                self.predict(x,False)
                
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
            updatedModel = CommitteeMember(self.parent, newInputs, newTargets)
        
        except RuntimeWarning as e:
            print('Error during Cholesky decomp for fantasy update. Fitting new model...')
            
            newInputs = torch.cat([self.train_x,x],dim=0)
            newTargets = torch.cat([self.train_y,y],dim=0)
            updatedModel = CommitteeMember(self.parent, newInputs, newTargets)
        
        #Update the data properties
        updatedModel.train_x = updatedModel.train_inputs[0]
        updatedModel.train_y = updatedModel.train_targets
        
        #Update parent's covar_module
        self.parent.covar_module = updatedModel.covar_module
        
        #Need to perform a prediction so that get_fantasy_model may be used to update later
        updatedModel.predict(x,False)
    
        return updatedModel
        