#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:33:59 2020

@author: nick
"""
import torch as t
import gpytorch
import matplotlib.pyplot as plt

class EpanechnikovKernel(gpytorch.kernels.Kernel):
    
    is_stationary = True
    has_lengthscale = True
    
    def __init__(self,**kwargs):
        
        super(EpanechnikovKernel,self).__init__()
        
    def forward(self, x1, x2, diag=False, **params):
        
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            dist = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, postprocess=True, **params)
            
            # get a mask of where the distance is sufficiently small to have non-zero covariance
            zeroMask = dist>1
            
            cov_mat = (1-dist)
            cov_mat[zeroMask] = 0
            
            return cov_mat

if __name__ == '__main__':
        
    # test the kernel on 1D data
    k = EpanechnikovKernel()
    
    nSteps = 100
    xr = t.linspace(-2,2,steps=nSteps)
    kr = k(xr,t.zeros(nSteps),diag=True)
    
    fig,ax = plt.subplots(1)
    ax.plot(xr,kr.detach())
    
    # test the kernel on 2D data with ARD
    k = EpanechnikovKernel(ard_num_dims=2)
    k.lengthscale = t.tensor([1,.5])
    
    yr = t.linspace(-2,2,steps=nSteps)
    xygrid = t.meshgrid([xr,yr])
    xy = t.stack(xygrid).permute([1,2,0]).reshape((nSteps*nSteps,2))
    
    kgrid = k(xy,t.zeros_like(xy),diag=True).reshape(nSteps,nSteps)
    
    fig,ax = plt.subplots(1)
    ax.contourf(xygrid[0],xygrid[1],kgrid.detach())