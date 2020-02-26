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

def makeModel(kernelClass,likelihood,w_gen):
    #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
    model = LocalGP.LocalGPModel(likelihood,kernelClass(ard_num_dims=2),w_gen=w_gen,inheritKernel=False,
                                 maxChildren=5)
    return model
    
def makeModels(kernelClass,likelihood,w_gen,k):
    models = []
    for i in range(k):
        models.append(makeModel(kernelClass, likelihood, w_gen))
    return models

def kFoldCrossValidation(kernelClass,likelihood,w_gen,k):
    #Create models for k-fold cross validation
    models = makeModels(kernel,likelihood,w_gen,k)
    
    predictions = []
    meanSquaredErrors = torch.zeros(size=(k,1))
    
    for index,model in zip(range(k),models):
        #Choose numSamples/k data points to withhold. Note k must divide numSamples.
        sliceStart = int(index*numSamples/k)
        sliceEnd = int((index+1)*numSamples/k)
        includedPointsIndices = list(range(numSamples))
        del includedPointsIndices[sliceStart:sliceEnd]
        
        #Train the model with 1/k th of the data withheld
        for randPairIndex in includedPointsIndices:
            randPair = randIndices[:,randPairIndex]
            x_train = xyGrid[randPair[0],randPair[1]].unsqueeze(0)
            y_train = z[randPair[0],randPair[1]]
            model.update(x_train,y_train)
    
        print('Done training model {0}'.format(index))    
    
        #Predict at withheld points for calculating out of bag MSE
        withheldPointsIndices = [index for index in range(numSamples) if index not in includedPointsIndices]
        randPairs = randIndices[:,withheldPointsIndices]
        randCoords = xyGrid[randPairs[0,:],randPairs[1,:]].unsqueeze(0)
        prediction = model.predict(randCoords)
        predictions.append(prediction)
        mse = torch.sum(torch.pow(prediction-z[randPairs[0,:],randPairs[1,:]],2),dim=list(range(prediction.dim())))/(numSamples-k)
        meanSquaredErrors[index] = mse
    
    del models
    return torch.mean(meanSquaredErrors)

#Construct a grid of input points
gridDims = 50
x,y = torch.meshgrid([torch.linspace(-1,1,gridDims), torch.linspace(-1,1,gridDims)])
xyGrid = torch.stack([x,y],dim=2).float()

#Evaluate a function to approximate
z = (5*torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2)+3*xyGrid[:,:,0]).reshape((gridDims,gridDims,1))

#Sample some random points, then fit a LocalGP model to the points
torch.manual_seed(6942069)
numSamples = 200
randIndices = torch.multinomial(torch.ones((2,gridDims)).float(),numSamples,replacement=True)

#Set # of models for cross-validation
k = 5
kernel = gpytorch.kernels.RBFKernel
likelihood = gpytorch.likelihoods.GaussianLikelihood

#Create different values of w_gen to test
w_genValues = torch.linspace(0,.6,8)
mse = torch.zeros(w_genValues.shape)
for i in range(w_genValues.shape[-1]):
    w_gen = w_genValues[i]
    print('w_gen={0}'.format(w_gen))
    mse[i] = kFoldCrossValidation(kernel,likelihood,w_gen,k)

