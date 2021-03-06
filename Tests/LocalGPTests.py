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
from math import inf

#Construct a grid of input points
gridDims = 50
x,y = torch.meshgrid([torch.linspace(-5,5,gridDims), torch.linspace(-5,5,gridDims)])
xyGrid = torch.stack([x,y],dim=2).double()

def evalModel(w_gen,numSamples,maxChildren0):
    #Set RNG seed
    torch.manual_seed(42069)
    
    #Evaluate a function to approximate, with added noise
    z = (torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2)).reshape((gridDims,gridDims,1))
    z += torch.randn(z.shape) * .05
    
    #Sample some random points, then fit a LocalGP model to the points
    #numSamples = 100
    randIndices = torch.multinomial(torch.ones((2,gridDims)).float(),numSamples,replacement=True)
        
    #Set # of models for cross-validation
    k = 5
    kernel = gpytorch.kernels.RBFKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood
    
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
    for randPairIndex in range(numSamples):
        randPair = randIndices[:,randPairIndex]
        x_train = xyGrid[randPair[0],randPair[1]].unsqueeze(0)
        y_train = z[randPair[0],randPair[1]]
        model.update(x_train,y_train)
        print(j)
        j += 1
    t1 = time.time()
    print('Done training')
    model.predict(xyGrid[randIndices[0,:],randIndices[1,:]])
    return t1-t0

maxChildrenVals = [inf]
numSamplesVals = [1000]
runtimes = {}
for maxChildren,numSamples in product(maxChildrenVals,numSamplesVals):
    runtimes[(maxChildren,numSamples)] = evalModel(.1,numSamples,maxChildren)

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