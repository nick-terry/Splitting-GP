# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:13:10 2020

@author: pnter
"""

'''
The purpose of this code is to experimentally
evaluate the usefulness/tightness of some different bounds on the GP
posterior variance.

We first consider the posterior variance bounce given by Theorem 3.1
in Lederer et al. (2019), "Posterior Variance Analysis of Gaussian Processes
with Application to Average Learning Curves". This bound assumes only Lipschitz
continuity of the kernel function, a relatively non-restrictive assumption.
'''
from varBoundFunctions import *
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
import numpy as np

#Load bikeshare data for testing
dataPath = 'C:/Users/pnter/Documents/GitHub/GP-Regression-Research/co2.csv'
data = pd.read_csv(dataPath)

#Remove years with missing data; remove year column, average column
data = data.loc[7:,(data.columns != 'Unnamed: 0')&(data.columns !='Average')]
#Flatten into 1D time series
data = data.to_numpy().flatten()

#Split data into testing and training
testTrainCutoff = int(np.ceil(data.shape[0]*.3))
dataTrain = data[:testTrainCutoff]
dataTest = data[testTrainCutoff:]

#Create GP model w/ Spectral Mixture kernel
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10,
                                                                   ard_num_dims=1)
        self.covar_module.initialize_from_data(train_x, train_y)
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

trainingPred = torch.tensor(range(testTrainCutoff)).float()
trainingResp = torch.tensor(dataTrain).float()

testPred = torch.tensor(range(data.shape[0])[testTrainCutoff:]).float()
testResp = torch.tensor(dataTest).float()
 
model = SpectralMixtureGPModel(trainingPred, trainingResp, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 300
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(trainingPred)
    loss = -mll(output, trainingResp)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

#Plot model fit at training data
model.eval()
likelihood.eval()

with torch.no_grad(),gpytorch.settings.fast_pred_var():
    fPredictions = likelihood(model(torch.tensor(range(data.shape[0])).float()))
    fMean = fPredictions.mean
    fVar = fPredictions.variance
    lower,upper = fPredictions.confidence_region()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(data.shape[0]),data,c='green')
    ax.plot(range(data.shape[0]),fMean,c='black')
    ax.fill_between(range(data.shape[0]), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([300,400])