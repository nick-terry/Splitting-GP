# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
s"""
import time
import LocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np
from itertools import product
import TestData

predictor,response = TestData.icethick()

def evalModel(w_gen,numSamples,maxChildren0):
    #Set RNG seed
    torch.manual_seed(42069)
    
    randIndices = torch.multinomial(torch.ones((1,predictor.shape[0])).float(),numSamples,replacement=True).squeeze(0)
        
    #Set # of models for cross-validation
    k = 5
    kernel = gpytorch.kernels.RBFKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood
    w_gen = .5
    
    def makeModel(kernelClass,likelihood,w_gen):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = LocalGP.LocalGPModel(likelihood,kernelClass(ard_num_dims=2),w_gen=w_gen,inheritKernel=True,
                                     maxChildren=maxChildren0)
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
    return t1-t0

numSamplesVals = [3000]
runtimes = {}
for maxChildren,numSamples in product(maxChildrenVals,numSamplesVals):
    runtimes[(maxChildren,numSamples)] = evalModel(.5,numSamples,maxChildren)

'''
#Predict over the whole grid for plotting
prediction = model.predict(xyGrid)

#Define a common scale for color mapping for contour plots
maxAbsVal = torch.max(torch.abs(z))
levels = np.linspace(-maxAbsVal,maxAbsVal,30)

#Plot true function
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,5))
contours = axes[0].contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),z.detach().squeeze(2),levels)
#plt.colorbar(contours)

#Plot GP regression approximation
axes[1].contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),prediction.detach().squeeze(2),levels)
#Show the points which were sampled to construct the GP model
sampledPoints  = xyGrid[randIndices[0,:],randIndices[1,:]]
axes[1].scatter(sampledPoints[:,0].detach(),sampledPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
childrenCenters = model.getCenters().squeeze(1)
axes[1].scatter(childrenCenters[:,0].detach(),childrenCenters[:,1].detach(),c='orange',s=24,edgecolors='white')
'''