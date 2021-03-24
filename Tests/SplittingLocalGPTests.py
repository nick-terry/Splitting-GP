# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import time
import LocalGP
import SplittingLocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from math import inf
import TestData

def getIcethick():
    predictor,response = TestData.icethick(scale=False)
    return predictor,response

def getKin40():
    predictor,response = TestData.kin40()
    return predictor,response

predictorsTrain,responseTrain,predictorsTest,responseTest = getKin40()


def evalModel(w_gen,numSamples):
    #Set RNG seed
    torch.manual_seed(42069)
    
    #Sample some random points, then fit a LocalGP model to the points
    #numSamples = 100
    randIndices = torch.multinomial(torch.ones((1,predictor.shape[0])).float(),numSamples,replacement=True).squeeze(0)
        
    #Set # of models for cross-validation
    k = 5
    kernel = gpytorch.kernels.RQKernel

    likelihood = gpytorch.likelihoods.GaussianLikelihood
    w_gen = .5
    
    
    def makeModel(kernelClass,likelihood,w_gen):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = SplittingLocalGP.SplittingLocalGPModel(likelihood,kernelClass(ard_num_dims=8),
                                                       splittingLimit=200,inheritKernel=True,inheritLikelihood=False)
        return model
        
    def makeModels(kernelClass,likelihood,w_gen,k):
        models = []
        for i in range(k):
            models.append(makeModel(kernelClass, likelihood, w_gen))
        return models
    
    model = makeModel(kernel,likelihood,w_gen=w_gen)
    t0 = time.time()
    j = 0
    for randSamp in range(numSamples):
        randIndex = randIndices[randSamp]
        x_train = predictor[randIndex].unsqueeze(0)
        y_train = response[randIndex].unsqueeze(0)
        model.update(x_train,y_train)
        print(j)
        j += 1
    t1 = time.time()
    print('Done training')
    return t1-t0,model,randIndices

numSamplesVals = [3000]
runtimes = {}
for numSamples in numSamplesVals:
    runtimes[numSamples] = evalModel(.5,numSamples)

numSamples = numSamplesVals[0]
model = runtimes[numSamples][1]
randIndices = runtimes[numSamples][2]

#Predict over the whole grid for plotting
prediction = model.predict(predictor)
totalPreds = prediction
prediction = totalPreds[0].detach()


mse = torch.mean((prediction.squeeze(-1)-response)**2)

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

#Show the points which were sampled to construct the GP model
'''
sampledPoints  = predictor[randIndices]
axes[1].scatter(sampledPoints[:,0].detach(),sampledPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
'''