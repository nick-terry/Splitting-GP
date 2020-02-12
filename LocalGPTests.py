# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
"""

import LocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt

#Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
likelihood = gpytorch.likelihoods.GaussianLikelihood
model = LocalGP.LocalGPModel(likelihood,kernel,w_gen=1)

#Construct a grid of input points
xyGrid = torch.stack(torch.meshgrid([torch.linspace(-1,1,25), torch.linspace(-1,1,25)]),dim=2).float()
#Evaluate a function to approximate
z = 5*torch.sin(xyGrid[:,:,0]**2+xyGrid[:,:,1]**2).reshape((25,25,1))
fig1 = plt.figure()
plt.contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),z.detach().squeeze(2))

#Add a new data pair {x,y}
model.update(xyGrid[0,0].unsqueeze(0),z[0,0])
#Predict at the data point just added
prediction = model.predict(xyGrid)
fig2 = plt.figure()
plt.contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),prediction.detach().squeeze(2))
#Update the existing child model
model.update(xyGrid[1,1].unsqueeze(0),z[1,1])