# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:01:27 2020

@author: pnter
"""

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

#Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
likelihood = gpytorch.likelihoods.GaussianLikelihood
model = LocalGP.LocalGPModel(likelihood,kernel,w_gen=.3,inheritKernel=False,numInducingPoints=5)

#Construct a grid of input points
gridDims = 25
x,y = torch.meshgrid([torch.linspace(-1,1,gridDims), torch.linspace(-1,1,gridDims)])
xyGrid = torch.stack([x,y],dim=2).float()

#Evaluate a function to approximate
z = 5*torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2).reshape((gridDims,gridDims,1))

#Sample some random points, then fit a LocalGP model to the points
torch.manual_seed(42069)
numSamples = 10
randIndices = torch.multinomial(torch.ones((2,gridDims)).float(),numSamples,replacement=True)

j = 0
for randPairIndex in range(numSamples):
    print(j)
    randPair = randIndices[:,randPairIndex]
    x_train = xyGrid[randPair[0],randPair[1]].unsqueeze(0)
    y_train = z[randPair[0],randPair[1]]
    model.update(x_train,y_train)
    j+=1
print('Done training')

#Predict over the whole grid for plotting
prediction = model.predict(xyGrid,M=5)

#Define a common scale for color mapping for contour plots
levels = np.linspace(-5,6,30)

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
