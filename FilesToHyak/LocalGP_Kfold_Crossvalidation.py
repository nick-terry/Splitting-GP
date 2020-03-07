# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
s"""

import LocalGP,SplittingLocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import math
import copy
import MemoryHelper

def makeModel(kernelClass,likelihood,w_gen):
    #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
    model = LocalGP.LocalGPModel(likelihood,kernelClass(ard_num_dims=2),w_gen=w_gen,inheritKernel=False)
    return model

def makeSplittingModel(kernelClass,likelihood,splittingLimit):
    #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
    model = SplittingLocalGP.SplittingLocalGPModel(likelihood,kernelClass(ard_num_dims=2),splittingLimit,inheritKernel=True)
    return model
    
def makeLocalGPModels(kernelClass,likelihood,w_gen,k):
    models = []
    for i in range(k):
        models.append(makeModel(kernelClass, likelihood, w_gen))
    return models

def makeSplittingLocalGPModels(kernelClass,likelihood,splittingLimit,k):
    models = []
    for i in range(k):
        models.append(makeSplittingModel(kernelClass, likelihood, splittingLimit))
    return models

def kFoldCrossValidation(kernelClass,likelihood,modelsList,numSamples):
    # of folds is equal to # of models given
    models = copy.deepcopy(modelsList)
    k = len(models)
    randIndices = completeRandIndices[:,:numSamples]
    
    predictions = []
    meanSquaredErrors = []
    elapsedTrainingTimes = []
    memoryUsages = []
    
    for index,model in zip(range(k),models):
        #Choose numSamples/k data points to withhold. Note k must divide numSamples.
        sliceStart = int(index*numSamples/k)
        sliceEnd = int((index+1)*numSamples/k)
        includedPointsIndices = list(range(numSamples))
        del includedPointsIndices[sliceStart:sliceEnd]
        
        t0 = time.time()
        
        #Train the model with 1/k th of the data withheld
        for randPairIndex in includedPointsIndices:
            randPair = randIndices[:,randPairIndex]
            x_train = xyGrid[randPair[0],randPair[1]].unsqueeze(0)
            y_train = z[randPair[0],randPair[1]]
            model.update(x_train,y_train)
        
        t1 = time.time()
        print('Done training model {0}'.format(index))    
    
        #Predict at withheld points for calculating out of bag MSE
        withheldPointsIndices = [index for index in range(numSamples) if index not in includedPointsIndices]
        randPairs = randIndices[:,withheldPointsIndices]
        randCoords = xyGrid[randPairs[0,:],randPairs[1,:]].unsqueeze(0)
        prediction = model.predict(randCoords)
        predictions.append(prediction)
        mse = torch.sum(torch.pow(prediction-z[randPairs[0,:],randPairs[1,:]],2),dim=list(range(prediction.dim())))/(numSamples/k)
       
        meanSquaredErrors.append(mse)
        elapsedTrainingTimes.append(t1-t0)
        memoryUsages.append(MemoryHelper.getMemoryUsage())
    
    del models
    return {'mse':meanSquaredErrors,'training_time':elapsedTrainingTimes,'memory_usage':memoryUsages}

def resultsToDF(results):
    for key in results:
        mse = results[key]['mse']
        mse = list(map(lambda x: float(x.detach()),mse))
        results[key] = {'mse':mse,'training_time':results[key]['training_time'],'memory_usage':results[key]['memory_usage']}
    
    df = pd.DataFrame(results).transpose()
    
    df['avg_mse'] = df['mse'].apply(np.mean)
    df['avg_training_time'] = df['training_time'].apply(np.mean)
    df['avg_training_time_per_update'] = df['avg_training_time']/df.index
    df['avg_memory_usage'] = df['memory_usage'].apply(np.mean)
    return df

#Construct a grid of input points
gridDims = 100
x,y = torch.meshgrid([torch.linspace(-1,1,gridDims), torch.linspace(-1,1,gridDims)])
xyGrid = torch.stack([x,y],dim=2).float()

#Evaluate a function to approximate
z = (5*torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2)+3*xyGrid[:,:,0]).reshape((gridDims,gridDims,1))

#Set # of folds for cross-validation
k = 5
kernel = gpytorch.kernels.RBFKernel
likelihood = gpytorch.likelihoods.GaussianLikelihood

#Create the models to be cross validated
modelsList = makeSplittingLocalGPModels(kernel,likelihood,500,k)

#Run k-fold cross validation for num samples ranging from 100-1200
maxSamples = 2500

#Sample some random points
torch.manual_seed(6942069)
completeRandIndices = torch.multinomial(torch.ones((2,gridDims)).float(),maxSamples,replacement=True)

numSamplesTensor = torch.linspace(100,maxSamples,maxSamples//100)
results = {}
for i in range(numSamplesTensor.shape[-1]):
    print('numSamples={0}'.format(100*(i+1)))
    results[int(numSamplesTensor[i])] = kFoldCrossValidation(kernel,likelihood,modelsList,int(numSamplesTensor[i]))

resultsDF = resultsToDF(results)
resultsDF.to_csv('experiment-results.csv')