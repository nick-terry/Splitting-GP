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
    
    #Center the matrix by subtracting centroid
    M0 = M - centroid.repeat((M.shape[0],1))

    #Scale columns by std
    stdCols = torch.std(M0,dim=0)
    M0 = M0/stdCols

    #Compute the low-rank Singular Value Decomposition of M0
    (U,S,V) = torch.svd(M0)
    
    #Take vector product of principal direction with rows of covar matrix
    M1 = torch.sum(M0 * V[:,0],dim=1)
    
    #Assign a cluster label based on product of the rows of the matrix with the principal direction
    labels = torch.where(M1>0,torch.ones(M1.shape[0]),torch.zeros(M1.shape[0]))
    
    #print('Split labels" {0}'.format(labels))
    return labels

'''
Compute one iteration of the Principal Direction Divisive Partitioning algorithm using power iteration to get principal direction

Arguments:
    
    M -- the matrix (torch tensor) on which the iteration will be performed
    
Returns:
    
    labels -- labels for each row of M which correspond to the clusters to which
        each row is assigned
'''
def pddp_piter(M):
    
    #Compute the centroid of the observed data
    centroid = torch.mean(M,dim=0)
    
    #Center the matrix by subtracting centroid
    M = M - centroid.repeat((M.shape[0],1))

    
    M0 = M.transpose(0,1).matmul(M)
    
    #Generate random unit vector to iterate on
    x = torch.rand((M.shape[1],1)).double()
    x0 = x
    
    x = M0.matmul(x)
    
    #Iterate until approximate convergence
    for i in range(100):
        if torch.sum(x0/torch.norm(x0)*x/torch.norm(x)) < 10**-3:
            break
            
        M0 = M0.transpose(0,1).matmul(M0)
        x = M0.matmul(x)
        x0 = x
        
    #Take vector product of principal direction with rows of matrix
    M1 = torch.sum(M*x.transpose(0,1),dim=1)
    
    #Assign a cluster label based on product of the rows of the matrix with the principal direction
    labels = torch.where(M1>0,torch.ones(M1.shape[0]),torch.zeros(M1.shape[0]))
    
    #print('Split labels" {0}'.format(labels))
    return labels


'''
Compute the Bayesian Information criterion for the given Likelihood and number of samples
'''
def BIC(L,n):
    return torch.log(n)-2*torch.log(L)

'''
Update the inverse covariance matrix cache using the Woodbury matrix identity.
It is assumed that both matrices have the same dimension.

Taking U=I and V=K-K_0, we get:
    
K^-1 = K_0^-1 - K_0^-1(I-(K-K_0)K_0^-1)^-1(K-K_0)K_0^-1

which simplifies to:
    
K^-1 = 2*K_0^-1 - K_0^-1*K*K_0^-1    

Arguments:
    
     K_0inv -- the inverse of the old covariance matrix (torch tensor)
     K -- the new covariance matrix (torch tensor)
    
Returns:
    
    Kinv -- the rank one update of the inverse of K
'''
def updateInverseCovarWoodbury(K_0inv,K):
    return 2*K_0inv - K_0inv.matmul(K).matmul(K_0inv)

'''
Compute the Haversine distance between two points on the Earth. Used to spatial regression
'''
def haversine(x,y):
    x,y = x.squeeze(0),y.squeeze(0)
    A = torch.sin((x[0]-y[0])/2)**2+torch.cos(x[0])*torch.cos(y[0])*torch.sin((x[1]-y[1])/2)**2
    C = 2*torch.atan2(A**.5, (1-A)**.5).unsqueeze(0).unsqueeze(1)
    return C*6371

'''
Create a function which projects a latitude and longitude to the Gnomomic projection at P
'''
def getGnomomic(P):
    P = P.squeeze()
    
    def project(x):
        x = x.squeeze()
        t1 = torch.sin(x[:,0])*torch.sin(P[0])
        t2 = torch.cos(P[0])*torch.cos(P[1]-x[:,1])
        cosC = t1 + torch.cos(x[:,0])*t2
        lat0 = torch.cos(P[1])*torch.sin(P[0]-x[:,0])/cosC
        lon0 = (t1-torch.sin(x[:,1])*t2)/cosC
        
        return torch.stack([lat0,lon0],dim=1)
    
    return project
        
        

        
        
    
    