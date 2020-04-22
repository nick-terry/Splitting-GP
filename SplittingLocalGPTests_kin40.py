# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import time
import LocalGP
import SplittingLocalGP,RegularGP
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from math import inf
import TestData
import pandas as pd

def getIcethick():
    predictor,response = TestData.icethick(scale=False)
    return predictor,response

def getKin40():
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.kin40()
    return predictorsTrain,responseTrain,predictorsTest,responseTest

predictorsTrain,responseTrain,predictorsTest,responseTest = getKin40()

def makeModel(kernelClass,likelihood,M,splittingLimit,inheritLikelihood):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = SplittingLocalGP.SplittingLocalGPModel(likelihood,kernelClass(ard_num_dims=8),
                                                       splittingLimit=splittingLimit,inheritKernel=True,
                                                       inheritLikelihood=True,
                                                       M=M,
                                                       mean=gpytorch.means.ZeroMean)
        return model

def makeRegularModel(kernelClass,likelihood):
        model = RegularGP.RegularGPModel(likelihood,kernelClass(ard_num_dims=8))
        return model
    
def makeModels(kernelClass,likelihood,w_gen,k):
    models = []
    for i in range(k):
        models.append(makeModel(kernelClass, likelihood, w_gen))
    return models

def evalModel(M=None,splittingLimit=500,inheritLikelihood=True,splitting=True):
    #Set RNG seed
    torch.manual_seed(42069)
        
    kernel = gpytorch.kernels.RBFKernel

    likelihood = gpytorch.likelihoods.GaussianLikelihood
    
    
    if splitting:
        model = makeModel(kernel,likelihood,M,splittingLimit,inheritLikelihood)
    else:
        model = makeRegularModel(kernel,likelihood)
        
    t0 = time.time()
    j = 0
    if splitting:
        for index in range(int(predictorsTrain.shape[0]))[::splittingLimit]:
            x_train = predictorsTrain[index:index+splittingLimit]
            y_train = responseTrain[index:index+splittingLimit]
            model.update(x_train,y_train)
            print(j)
            j += 1
    else:
        model.update(predictorsTrain,responseTrain)

    t1 = time.time()
    print('Done training')
    
    '''
    #Predict over the whole grid for plotting
    totalPreds = model.predict(predictorsTest,individualPredictions=False)
    prediction = totalPreds[0].detach()
    '''
    
    return model

def evalgptModel():
    train_x = predictorsTrain
    train_y = responseTrain
    class ExactGPModel(gpytorch.models.ExactGP):
        
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = ExactGPModel(train_x, train_y, likelihood)
    likelihood = model.likelihood
    model.train()
    likelihood.train()
    model.train_inputs = (torch.cat([model.train_inputs[0], predictorsTrain]),)
    model.train_targets = torch.cat([model.train_targets, responseTrain])
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.train_inputs[0])
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model


'''
model = evalgptModel()

model.eval()
model.likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = model.likelihood(model(predictorsTest))
    
rmse = torch.sqrt(torch.mean((preds.mean-responseTest)**2))
'''

paramsList = [{'M':None,'splittingLimit':50,'inheritLikelihood':True,'splitting':True},
              {'M':None,'splittingLimit':125,'inheritLikelihood':True,'splitting':True}.
              {'M':None,'splittingLimit':250,'inheritLikelihood':True,'splitting':True},
              {'M':None,'splittingLimit':500,'inheritLikelihood':True,'splitting':True},
              {'M':None,'splittingLimit':1000,'inheritLikelihood':True,'splitting':True},
              {'M':None,'splittingLimit':2000,'inheritLikelihood':True,'splitting':True},
              {'M':None,'splittingLimit':10000,'inheritLikelihood':True,'splitting':True}]

resultsArr = torch.zeros((len(paramsList),1))

for i in range(len(paramsList)):
    model = evalModel(**paramsList[i])    
    preds = model.predict(predictorsTest)
    rmse = torch.sqrt(torch.mean((preds-responseTest)**2))
    resultsArr[i] = rmse

df = pd.DataFrame()
df['params'] = paramsList
df['rmse'] = resultsArr.detach().numpy()
df.to_csv('kin40_results_splitting.csv')

'''
#Define a common scale for color mapping for contour plots
maxAbsVal = torch.max(torch.abs(response))
levels = np.linspace(-maxAbsVal,maxAbsVal,30)

#Plot true function
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,5))
contours = axes[0].scatter(predictor[:,0].detach(),predictor[:,1].detach(),c=response)

#Plot GP regression approximation
axes[1].scatter(predictor[:,0].detach(),predictor[:,1].detach(),c=prediction.squeeze(-1))

childrenCenters = model.getCenters().squeeze(1)
axes[1].scatter(childrenCenters[:,0].detach(),childrenCenters[:,1].detach(),c='orange',s=24,edgecolors='white')
'''
#Show the points which were sampled to construct the GP model
'''
sampledPoints  = predictor[randIndices]
axes[1].scatter(sampledPoints[:,0].detach(),sampledPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
'''