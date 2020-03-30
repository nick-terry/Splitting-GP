# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:58:59 2020

@author: pnter
"""

import torch

'''
Compute one iteration of the Principal Direction Divisive Partitioning algorithm.

Arguments:
    
    M -- the matrix (torch tensor) on which the iteration will be performed
    
Returns:
    
    labels -- labels for each row of M which correspond to the clusters to which
        each row is assigned
'''
def pddp(M):
    
    #Compute the centroid of the observed data
    centroid = torch.mean(M,dim=0)
    
    #Compute covariance matrix M0 by subtracting the centroid from the rows of M
    M0 = M - centroid.repeat((M.shape[0],1))
    
    #Compute the Singular Value Decomposition of M0
    (U,S,V) = torch.svd(M0.T)
    
    #Take vector product of principal direction with rows of covar matrix
    M1 = torch.sum(M0 * U[0],dim=1)
    
    #Assign a cluster label based on product of the rows of the covar matrix with the principal direction
    labels = torch.where(M1>0,torch.ones(M1.shape[0]),torch.zeros(M1.shape[0]))
    
    #print('Split labels" {0}'.format(labels))
    return labels

'''
Update the inverse covariance matrix cache using the Woodbury matrix identity.
It is assumed that both matrices have the same dimension.

Taking U=I and V=K-K_0, we get:
    
K^-1 = K_0^-1 - K_0^-1(I-(K-K_0)K_0^-1)^-1(K-K_0)K_0^-1

which simplifies to:
    
K^-1 = K_0^-1*K*K_0^-1    

Arguments:
    
     K_0inv -- the inverse of the old covariance matrix (torch tensor)
     K -- the new covariance matrix (torch tensor)
    
Returns:
    
    Kinv -- the rank one update of the inverse of K
'''
def updateInverseCovarWoodbury(K_0inv,K):
    return K_0inv.matmul(K).matmul(K_0inv)