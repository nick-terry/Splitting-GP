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

#Load bikeshare data for testing
dataPath = 'C:/Users/pnter/Documents/GitHub/GP-Regression-Research/Bike-Sharing-Dataset/hour.csv'
data = pd.read_csv(dataPath)

data1mo = data.loc[data.instant<=24*30,:]
dataExtrap = data.loc[data.instant<=24*30+72,:]
dataTest = data.loc[(data.instant>24*30)&(data.instant<=24*30+72),:]
plt.plot(data1mo.instant,data1mo.cnt)

#Create GP model w/ Spectral Mixture kernel
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10,
                                                                   ard_num_dims=train_x.shape[1])
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

#Fit GP regression
predictors = ['instant', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
response = 'cnt'

trainingPred = torch.tensor(data1mo.loc[:,predictors].values).float()
trainingResp = torch.tensor(data1mo.loc[:,'cnt'].values).float()

testPred = torch.tensor(dataTest.loc[:,predictors].values).float()
testResp = torch.tensor(dataTest.loc[:,'cnt'].values).float()

extrapPred = torch.tensor(dataExtrap.loc[:,predictors].values).float()
extrapResp = torch.tensor(dataExtrap.loc[:,'cnt'].values).float()

 
model = SpectralMixtureGPModel(trainingPred, trainingResp, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 1000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(trainingPred)
    loss = -mll(output, trainingResp)
    loss.sum().backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.sum()))
    optimizer.step()

#Plot model fit at training data
model.eval()
likelihood.eval()

with torch.no_grad(),gpytorch.settings.fast_pred_var():
    fPredictions = likelihood(model(trainingPred))
    fMean = fPredictions.mean
    fVar = fPredictions.variance
    lower,upper = fPredictions.confidence_region()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(trainingPred[:,0],trainingResp)
    ax.plot(trainingPred[:,0],fMean,c='black')
    ax.fill_between(trainingPred[:,0].numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    
    fPredictionsTest = likelihood(model(extrapPred))
    fMeanTest = fPredictionsTest.mean
    fVarTest = fPredictionsTest.variance
    lower,upper = fPredictionsTest.confidence_region()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(extrapPred[:,0],extrapResp)
    ax2.plot(extrapPred[:,0],fMeanTest,c='black')
    ax2.fill_between(extrapPred[:,0].numpy(), lower.numpy(), upper.numpy(), alpha=0.5)