# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:50:23 2020

@author: pnter
"""

import torch
import gpytorch
import pickle


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.likelihood = likelihood
        
        self.initTraining()
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def initTraining(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        #Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        #Perform training iterations
        for i in range(200):
            self.optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            self.optimizer.step()
   
    def predict(self,x):
        #Switch to eval/prediction mode
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self(x))
        return prediction
    
    
train_x = [torch.tensor([[ 0.3878, -0.1837],
        [ 0.3878, -0.1020],
        [ 1.0000,  0.6735]])]
train_y = torch.tensor([0.2710, 0.2042, 0.3384])

kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = GPModel(train_x, train_y, likelihood, kernel)

with open('error_state_dict','rb') as f:
    err_state_dict = pickle.load(f)

model.load_state_dict(err_state_dict)

model.predict(model.train_inputs[0].unsqueeze(0))

fantasy_x1 = torch.tensor([[ 0.3061, -0.2245]])
fantasy_y1 = torch.tensor([0.2633])

model = model.get_fantasy_model(fantasy_x1,fantasy_y1)

