# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
"""

import LocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt

kernel = gpytorch.kernels.RBFKernel()
likelihood = gpytorch.likelihoods.GaussianLikelihood
model = LocalGP.LocalGPModel(likelihood,kernel,w_gen=1)

xv, yv = torch.meshgrid([torch.linspace(-1,1,25), torch.linspace(-1,1,25)])
xv, yv = xv.float(),yv.float()
z = torch.sin(xv**2+yv**2)
plt.contourf(xv,yv,z)

model.update(torch.tensor([xv[0,0],yv[0,0]]),z[0,0])
model.update(torch.tensor([xv[1,1],yv[1,1]]),z[1,1])

ind,dist = model.getClosestChild(torch.tensor([xv[1,1],yv[1,1]]))