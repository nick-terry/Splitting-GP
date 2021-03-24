#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:35:46 2020

@author: nick
"""

import torch
import gpytorch
from extra_kernels import EpanechnikovKernel

class VariationalSplittingGPModel(gpytorch.models.ApproximateGP):
    
    def __init__(self, train_x, train_y, child_models):
        
        var_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.shape[0])
        var_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(self,
                        inducing_points=train_x,
                        variational_distribution=var_distribution,
                        learn_inducing_locations=False)
        
        
        super(VariationalSplittingGPModel,self).__init__(var_strategy)
        
        self.child_train_list = [child.train_x for child in child_models]
        self.child_models = child_models
        
        # register the parameters of the child models
        for i,child in enumerate(self.child_models):
            self.__setattr__('child_model_{}'.format(i), child)
        
        # self.parent_kernel = EpanechnikovKernel(ard_num_dims=1)
        self.parent_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
        
        self.mean_module = CompositeMean(self.child_models, self.parent_kernel)
        # self.covar_module = CompositeCovar(self.child_models, self.parent_kernel)
        
    def forward(self,x):
        
        mean_x = self.mean_module(x)
        # covar_x = self.covar_module(x)
        covar_x = self.parent_kernel(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class CompositeMean(gpytorch.means.Mean):
    
    def __init__(self,children,parent_kernel):
        
        super().__init__()
        
        self.child_models = children
        self.x_train_list = [child.train_x for child in children]
        self.parent_kernel = parent_kernel
        
    def forward(self,x):
        
        yHatArr = torch.zeros((len(self.child_models),x.shape[0]))
        # compute predictions with the child models
        for i,child in enumerate(self.child_models):
            yHatArr[i] = child(x).mean
    
        # get probability that data belongs to each child model
        PArr = self.getLatentProbs(x)
        
        # take expectation over latent variable to get posterior mean
        yHat = torch.sum(PArr * yHatArr,axis=0)
        
        return yHat
    
    def getLatentProbs(self,x):
        """
        Compute the kernel density estimates of the probability that each x
        belongs to each local model.

        Parameters
        ----------
        x : torch tensor
            The inputs.

        Returns
        -------
        PArr : torch tensor
            The KDE of probabilities.

        """
        
        PArr = torch.zeros((len(self.child_models),x.shape[0]))
        # compute covar and sum over all data assigned to the child model
        for i,model in enumerate(self.child_models):
            PArr[i] = torch.sum(self.parent_kernel(x,self.x_train_list[i]).evaluate(),dim=1)
        
        # normalize the summed covar to get KDE
        PArr = PArr / torch.sum(PArr,axis=0)
        
        return PArr
    
class CompositeCovar(gpytorch.module.Module):
    
    def __init__(self,children,parent_kernel,**kwargs):
        
        super(CompositeCovar,self).__init__(**kwargs)
        
        self.child_models = children
        self.x_train_list = [child.train_x for child in children]
        self.parent_kernel = parent_kernel
        
    def forward(self,x):
        
        kHatArr = torch.zeros((len(self.child_models),x.shape[0]))
        
        # compute covar with the child models
        for i,child in enumerate(self.child_models):
            kHatArr[i] = child(x).variance
    
        # get probability that data belongs to each child model
        PArr = self.getLatentProbs(x)
        
        # take expectation over latent variable to get posterior variance for each x
        # i.e. we are using diagonal covar structure
        kHat = torch.sum(PArr * kHatArr,axis=0)
        
        return torch.diag(kHat)
    
    def getLatentProbs(self,x):
        """
        Compute the kernel density estimates of the probability that each x
        belongs to each local model.

        Parameters
        ----------
        x : torch tensor
            The inputs.

        Returns
        -------
        PArr : torch tensor
            The KDE of probabilities.

        """
        
        PArr = torch.zeros((len(self.child_models),x.shape[0]))
        # compute covar and sum over all data assigned to the child model
        for i,model in enumerate(self.child_models):
            PArr[i] = torch.sum(self.parent_kernel(x,self.x_train_list[i]).evaluate(),dim=1)
        
        # normalize the summed covar to get KDE
        PArr = PArr / torch.sum(PArr,axis=0)
        
        return PArr