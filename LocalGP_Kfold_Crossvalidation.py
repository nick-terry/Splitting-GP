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

def resultsToDF(results,modelType,replicate,params):
    for key in results:
        mse = results[key]['mse']
        mse = list(map(lambda x: float(x.detach()),mse))
        results[key] = {'mse':mse,'training_time':results[key]['training_time'],'memory_usage':results[key]['memory_usage']}
    
    df = pd.DataFrame(results).transpose()
    
    df['avg_mse'] = df['mse'].apply(np.mean)
    df['avg_training_time'] = df['training_time'].apply(np.mean)
    df['avg_training_time_per_update'] = df['avg_training_time']/df.index
    df['avg_memory_usage'] = df['memory_usage'].apply(np.mean)
    df['model'] = modelType
    paramName = 'splittingLimit' if modelType == 'splitting' else 'w_gen'
    df['params'] = '{0}={1}'.format((paramName, params[paramName]))
    df['replicate'] = replicate
    
    return df

'''
Create a new experiment for analyzing a GP model.

Arguments:

    modelType: A string giving the type of GP model to test. Must be one of the following: 'splitting','local','exact'

Keyword Arguments:
    
    params: A dict which specifies the model hyperparameters. For a splitting model, this is 'splittingLimit'.
        For a local model, this is 'w_gen'. Exact models require no hyperparameters.
        
    folds: # of folds in the crossvalidation. Must be an int greater than 1. Don't make this too big. Default is 5.
        
    gridDims: # of grid points used for evaluating the response function of the experiment, in one dimension.
        Note that we need gridDims**n>=maxSamples. In practice, we want gridDims**n>2*maxSamples for the
        experiment to give useful results. By default, this will be 100.
        
    maxSamples: Maximum number of samples to run the model against. Should be a multiple of 100. In the experiment,
        we will run the model with numbers of samples ranging from 100 to maxSamples in increments of 100.
        
    replications: # of times to replicate this experiment. This can be increased to produce more accurate estimates
        of mean model performance. A difference RNG seed will be used for each replication.
        
'''
def runCrossvalidationExperiment(modelType,**kwargs):
    #Verify that the modelType/params combination is valid
    assert (modelType=='exact' and 'params' not in kwargs) or ...
    (modelType=='splitting' and 'params' in kwargs and 'splittingLimit' in kwargs['params']) or ...
    (modelType=='local' and 'params' in kwargs and 'w_gen' in kwargs['params'])
    
    params = kwargs['params']
    kernel = gpytorch.kernels.RBFKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood
        
    #Construct a grid of input points
    gridDims = kwargs['gridDims'] if 'gridDims' in kwargs else 100
    x,y = torch.meshgrid([torch.linspace(-1,1,gridDims), torch.linspace(-1,1,gridDims)])
    xyGrid = torch.stack([x,y],dim=2).float()
    
    #Evaluate a function to approximate
    z = (5*torch.sin(xyGrid[:,:,0]**2+(2*xyGrid[:,:,1])**2)+3*xyGrid[:,:,0]).reshape((gridDims,gridDims,1))
    
    #Set # of folds for cross-validation
    k = kwargs['folds'] if 'folds' in kwargs else 5
    
    #Run k-fold cross validation for num samples ranging from 100-maxSamples
    maxSamples = kwargs['maxSamples'] if 'maxSamples' in kwargs else 2500
    
    #Set # of replications to perform
    replications = kwargs['replications'] if 'replications' in kwargs else 30
    
    #Generate a seed for each replicate
    torch.manual_seed(6942069)
    seeds = torch.floor(100000*torch.rand(size=(replications,1)))
    
    experimentDF = pd.DataFrame()
    
    #Create a function to get new models for each step of the experiment
    if modelType=='exact':
        def getModelsFunction(): return makeLocalGPModels(kernel, likelihood, 0, k)
    elif modelType=='local':
        def getModelsFunction(): return makeLocalGPModels(kernel, likelihood, params['w_gen'], k)
    elif modelType=='splitting':
        def getModelsFunction(): return makeSplittingLocalGPModels(kernel, likelihood, params['splittingLimit'], k)
    
    for replicate in range(replications):
        
        #Create the models to be cross validated
        modelsList = getModelsFunction()
        
        #Set the seed for the RNG
        torch.manual_seed(int(seeds[replicate]))
        
        #Sample some random points
        completeRandIndices = torch.multinomial(torch.ones((2,gridDims)).float(),maxSamples,replacement=True)
        
        numSamplesTensor = torch.linspace(100,maxSamples,maxSamples//100)
        results = {}
        for i in range(numSamplesTensor.shape[-1]):
            print('numSamples={0}'.format(100*(i+1)))
            results[int(numSamplesTensor[i])] = kFoldCrossValidation(kernel,likelihood,modelsList,int(numSamplesTensor[i]))
    
        resultsDF = resultsToDF(results,modelType,replicate,params)
        experimentDF.merge(resultsDF,how='outer')
        
    return experimentDF