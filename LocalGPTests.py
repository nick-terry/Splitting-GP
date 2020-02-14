# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
s"""

import LocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

#Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
likelihood = gpytorch.likelihoods.GaussianLikelihood
model = LocalGP.LocalGPModel(likelihood,kernel,w_gen=.3,inheritKernel=False)

#Construct a grid of input points
gridDims = 25
x,y = torch.meshgrid([torch.linspace(-1,1,gridDims), torch.linspace(-1,1,gridDims)])
xyGrid = torch.stack([x,y],dim=2).float()

#Evaluate a function to approximate
z = 5*torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2).reshape((gridDims,gridDims,1))

#Sample some random points, then fit a LocalGP model to the points
torch.manual_seed(42069)
numSamples = 200
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

'''
#Add a new data pair {x,y}
model.update(xyGrid[0,0].unsqueeze(0),z[0,0])

prediction = model.predict(xyGrid[0,0])

#Update the existing child model
model.update(xyGrid[9,12].unsqueeze(0),z[9,12])

model.update(xyGrid[20,3].unsqueeze(0),z[20,3])
'''

#Predict over the whole grid for plotting
prediction = model.predict(xyGrid,M=5)

#Define a common scale for color mapping for contour plots
levels = np.linspace(-5,5,30)

#Plot true function
fig1 = plt.figure()
plt.contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),z.detach().squeeze(2),levels)
plt.colorbar()

#Plot GP regression approximation
fig2 = plt.figure()
plt.contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),prediction.detach().squeeze(2),levels)
plt.colorbar()
#Show the points which were sampled to construct the GP model
sampledPoints  = xyGrid[randIndices[0,:],randIndices[1,:]]
plt.scatter(sampledPoints[:,0].detach(),sampledPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
childrenCenters = model.getCenters().squeeze(1)
plt.scatter(childrenCenters[:,0].detach(),childrenCenters[:,1].detach())