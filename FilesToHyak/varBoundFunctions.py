# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:13:10 2020

@author: pnter
"""

'''
The purpose of this code is to experimentally
evaluate the usefulness/tightness of some different bounds on the GP
posterior variance.

We first consider the posterior variance bounce given by Theorem 3.1
in Lederer et al. (2019), "Posterior Variance Analysis of Gaussian Processes
with Application to Average Learning Curves". This bound assumes only Lipschitz
continuity of the kernel function, a relatively non-restrictive assumption.
'''

import numpy as np
import torch
import warnings
warnings.simplefilter('error', RuntimeWarning)
'''
Compute the number of data points in D whose Euclidean distance from x is
less than rho
'''
def size_B_rho(x,D,rho):
    return len(list(filter(lambda xPrime:np.linalg.norm(x-xPrime)<=rho,D)))

'''
Compute the Euclidean distance from x to the closest data point in D
'''
def minDist(x,D):
    return np.min(np.array(list(map(lambda xPrime:np.linalg.norm(x-xPrime),D))))

'''
Compute the Euclidean distance from x to the centroid of D
'''
def centroidDist(x,D):
    return np.linalg.norm(x-np.mean(D))

'''
Compute the sum of Euclidean distances from x to points in D, divided by size of D
'''
def sumDist(x,D):
    return np.sum(np.array(list(map(lambda xPrime:np.linalg.norm(x-xPrime),D))))/D.shape[0]

'''
Given an n by n symmetric and PSD matrix A, compute variational bounds on the
entries of its inverse B using result from Robinson and Wathen (1992).

Returns: L,U which give lower and upper bounds on the entries of A, respectively
'''
def variationalBounds(A):
    #Pre-compute S, which is a needed value for the matrix approximation
    #i is row of A, j is column of A
    def s(i,j):
         return np.sum(np.multiply(A[i,:],A[:,j]))
    global S
    S = np.zeros(A.shape)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            S[i,j] = s(i,j)
            
    #Pre-compute the first and last eigenvalues of A, which are alpha and beta,
    #respectively
    eigs = np.linalg.eigvals(A)
    alpha,beta = eigs[-1],eigs[0]
    
    #Initialize L,U
    L,U = np.zeros(A.shape),np.zeros(A.shape)
    
    global invExpr1
    global invExpr2
    #Compute bounds for the diagonal entries of B using Theorem 3.16
    for i in range(n):
        for j in range(n):
            if i==j:
                #Compute bounds for the diagonal entries of B using Theorem 3.16
                L[i,i] = 1/alpha+(alpha-A[i,i])**2/(alpha*(alpha*A[i,i]-S[i,i]))
                U[i,i] = 1/beta-(A[i,i]-beta)**2/(beta*(S[i,i]-beta*A[i,i]))
            else:
                #Compute bounds for the off-diagonal entries of B using Theorem 3.29
                #Precompute parts of the expression for efficiency
                global line5,line6_l,line6_u
                line1 = 1/S[i,i]/S[j,j]*((A[i,i]*A[j,j]*S[j,j]-A[i,i]*A[j,i]*S[j,j]-A[j,j]*A[j,i]*S[i,i])*(1/beta+1/alpha)/2-A[i,i]*A[j,j]*A[j,i])
                invExpr = 1/(-A[j,i]**2+2*A[j,i]*S[i,j]*(1/beta+1/alpha)/2+S[i,i]*S[j,j]*(1/beta-1/alpha)**2/4-S[i,j]**2*(1/beta-1/alpha)**2/4)
                line3 = (-A[j,i]+S[i,j]*(1/beta+1/alpha)/2)*(-A[j,j]*A[j,i]+(A[j,j]*S[i,j]-A[j,i]*S[j,j])*(1/beta+1/alpha)/2)
                line4_p1 = (-A[j,i]*A[i,i]*(A[i,i]*S[i,j]-A[j,i]*S[i,i])*(1/beta+1/alpha)/2)
                line4_p2 = (1/beta-1/alpha)/2
                line5 = -A[j,i]**2+2*A[j,i]*(S[i,j]-A[j,i]*A[i,i])*(1/beta+1/alpha)/2+S[j,j]*(S[i,i]-A[j,j]**2)*(1/beta-1/alpha)**2/4
                line6_l = (2*A[i,i]*A[j,i]*S[i,j]-S[i,j]**2-A[j,i]**2*S[i,i])*(1/beta+1/alpha)**2/4
                #Believe there is a typo in the paper for this line of the expression (paper claims first S[i,j] should be A[i,j])
                line6_u = (2*A[i,i]*A[j,i]*S[i,j]-S[i,j]**2-A[j,i]**2*S[i,i])*(1/beta+1/alpha)**2/4
                line7 = np.sqrt(-A[j,i]**2+2*A[j,i]*(S[i,j]-A[j,i]*A[j,j])*(1/beta+1/alpha)/2+S[i,i]*(S[j,j]-A[j,j]**2)*(1/beta-1/alpha)**2/4+\
                     (2*A[j,j]*A[j,i]*S[i,j]-S[i,j]**2-A[j,i]**2*S[j,j])*(1/beta+1/alpha)**2/4)
                         
                L[j,i] = line1+\
                    invExpr*\
                    (line3*\
                     line4_p1-line4_p2*\
                     np.sqrt(line5+\
                     line6_l)*\
                    line7)
                
                U[j,i] = line1+\
                    invExpr*\
                    (line3*\
                     line4_p1+line4_p2*\
                     np.sqrt(line5+\
                     line6_u)*\
                     line7)
    return L,U
'''
Creates a parameterized function which gives an upper bound on the posterior
variance, assuming only Lipschitz continuity of the kernel

noise: The noise (variance) of the training data

D: The set of all training data for the GP

rho: The "Information radius" i.e. radius of the ball centered on x within
which training data is considered to be informative

K: The kernel function

L: The Lipschitz constant of the kernel function K()

Returns: A function which gives an upper bound on posterior covariance for 
predictions
'''
def getPostVarBoundFnLipschitz(D, noise, rho, K, L):
    numConstant = (4*L*rho-L**2*rho**2)
    
    '''
    Returns an upper bound on the posterior variance of the GP regression
    
    x: The point of prediction for which the upper bound on the posterior
    variance holds
    '''
    def postVarBound(x):
        #Compute the size of the subset of training data within 
        #the open ball B_rho(x)
        const_size_B_rho = size_B_rho(x,D,rho)
        if type(x) is np.ndarray:
            numpy_x = x
            torch_x = torch.tensor(x)
        elif type(x) is torch.tensor:
            numpy_x = x.numpy()
            torch_x = x
        else:
            numpy_x = np.array(x)
            torch_x = torch.tensor(x)
            
        return (numConstant*const_size_B_rho*K(torch_x,torch_x)+noise*K(torch_x,torch_x))/(const_size_B_rho*(K(torch_x,torch_x)+2*L*rho)+noise)
    
    return postVarBound

'''
Creates a parameterized function which gives a (tighter) upper bound on the
posterior variance, assuming the kernel is isotropic and decreasing. Bound is
given by Corollary 3.1 in Lederer et al. (2019).

noise: The noise (variance) of the training data

D: The set of all training data for the GP

rho: The "Information radius" i.e. radius of the ball centered on x within
which training data is considered to be informative

K: The kernel function, which depends only on the distance d between 2 points

Returns: A function which gives an upper bound on posterior covariance for 
predictions
'''       
def getPostVarBoundFn(D, noise, rho, K):
    inputs = torch.tensor([0, rho])
    outputs = K(inputs)
    K_0 = outputs[0,0]
    K_rho_sq = outputs[0,1]
    print(K_0)
    print(K_rho_sq)
    '''
    Returns an upper bound on the posterior variance of the GP regression
    
    x: The point of prediction for which the upper bound on the posterior
    variance holds
    '''
    def postVarBound(x):
        
        if type(x) is np.ndarray:
            numpy_x = x
            torch_x = torch.tensor(x)
        elif type(x) is torch.tensor:
            numpy_x = x.numpy()
            torch_x = x
        else:
            numpy_x = np.array(x)
            torch_x = torch.tensor(x)
            
        const_size_B_rho = size_B_rho(x,D,rho)
        return K_0-K_rho_sq*const_size_B_rho/(K_0*const_size_B_rho+noise)
        
    return postVarBound

def getPostVarBoundFnTest(D, noise, rho, K):
    inputs = torch.tensor([0, rho])
    outputs = K(inputs)
    print(outputs)
    K_0 = outputs[0,0]
    print(K_0)
    '''
    Returns an upper bound on the posterior variance of the GP regression
    
    x: The point of prediction for which the upper bound on the posterior
    variance holds
    '''
    def postVarBound(x):
        
        if type(x) is np.ndarray:
            numpy_x = x
            torch_x = torch.tensor(x)
        elif type(x) is torch.tensor:
            numpy_x = x.numpy()
            torch_x = x
        else:
            numpy_x = np.array(x)
            torch_x = torch.tensor(x)
            
        return K_0-K_0/(minDist(x,D)+1)
        
    return postVarBound

#Test the functions here
def tests():
    D = np.linspace(0,10,11)
    rho = 3
    
    inputs = [10,5]
    correctOutputs = [4,7]
    results = []
    
    for inputVal in inputs:
        results.append(size_B_rho(inputVal,D,rho))

    try:
        for result,correctOutput in zip(results,correctOutputs):
            assert result==correctOutput
    except:
        print('size_B_rho returned {0}, correct value is {1}'.format(result,correctOutput))
    
    #Test variational matrix approximations
    A = np.array([[1,2],[2,1]])
    L,U = variationalBounds(A)
    try:
        assert L<=A
        assert U>=A
    except:
        print('variationalBounds returned incorrect bounds for matrix {0}'.format(A))
    
    #Generate a few random symmetric PSD matrices to test the function
    A = np.random.randint(0,100+1,size=(4,4))
    A = (A+A.T)/2
    print(A)
    L,U = variationalBounds(A)
    try:
        assert L<=A
        assert U>=A
    except:
        print('variationalBounds returned incorrect bounds for matrix {0}'.format(A))
        
    A = np.random.randint(0,100+1,size=(8,8))
    A = (A+A.T)/2
    L,U = variationalBounds(A)
    try:
        assert L<=A
        assert U>=A
    except:
        print('variationalBounds returned incorrect bounds for matrix {0}'.format(A))