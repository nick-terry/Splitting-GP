# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
s"""

import LocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np


#Construct a grid of input points
gridDims = 1000
x,y = torch.meshgrid([torch.linspace(-5,5,gridDims), torch.linspace(-5,5,gridDims)])
xyGrid = torch.stack([x,y],dim=2).float()

#Evaluate a function to approximate
z = (5*torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2)+3*xyGrid[:,:,0]).reshape((gridDims,gridDims,1))

#Sample some random points, then fit a LocalGP model to the points
torch.manual_seed(42069)
numSamples = 500
randIndices = torch.multinomial(torch.ones((2,gridDims)).float(),numSamples,replacement=True)
    
#Set # of models for cross-validation
k = 5
kernel = gpytorch.kernels.RBFKernel
likelihood = gpytorch.likelihoods.GaussianLikelihood
w_gen = 0

def makeModel(kernelClass,likelihood,w_gen):
    #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
    model = LocalGP.LocalGPModel(likelihood,kernelClass(ard_num_dims=2),w_gen=w_gen,inheritKernel=True)
    return model
    
def makeModels(kernelClass,likelihood,w_gen,k):
    models = []
    for i in range(k):
        models.append(makeModel(kernelClass, likelihood, w_gen))
    return models

model = makeModel(kernel,likelihood,w_gen=0)
for randPairIndex in range(numSamples):
    randPair = randIndices[:,randPairIndex]
    x_train = xyGrid[randPair[0],randPair[1]].unsqueeze(0)
    y_train = z[randPair[0],randPair[1]]
    model.update(x_train,y_train)

print('Done training')    


#Predict over the whole grid for plotting
prediction = model.predict(xyGrid)

#Define a common scale for color mapping for contour plots
maxAbsVal = torch.max(torch.abs(z))
levels = np.linspace(-z,z,30)

#Plot true function
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,5))
contours = axes[0].contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),z.detach().squeeze(2),levels)
#plt.colorbar(contours)

#Plot GP regression approximation
axes[1].contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),prediction.detach().squeeze(2),levels)
#Show the points which were sampled to construct the GP model
sampledPoints  = xyGrid[randIndices[0,:],randIndices[1,:j]]
axes[1].scatter(sampledPoints[:,0].detach(),sampledPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
childrenCenters = model.getCenters().squeeze(1)
axes[1].scatter(childrenCenters[:,0].detach(),childrenCenters[:,1].detach(),c='orange',s=24,edgecolors='white')

