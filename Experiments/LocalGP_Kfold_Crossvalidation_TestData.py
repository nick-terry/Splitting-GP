# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
s"""

import RegularGP,LocalGP,SplittingLocalGP
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import math
import copy
import MemoryHelper
import multiprocessing as mp
import ExperimentProcessingPool
import GPLogger
import time
import TestData

#Set some gpytorch settings
gpytorch.settings.fast_computations.covar_root_decomposition.on()

def makeExactModel(kernelClass,likelihood,inheritKernel=True,fantasyUpdate=True):
    return RegularGP.RegularGPModel(likelihood,kernelClass(ard_num_dims=2),inheritKernel,fantasyUpdate)

def makeExactModels(kernelClass,likelihood,k,inheritKernel=True,fantasyUpdate=True):
    models = []
    for i in range(k):
        models.append(makeExactModel(kernelClass,likelihood,inheritKernel,fantasyUpdate))
    return models

def makeLocalModel(kernelClass,likelihood,w_gen,**kwargs):
    if 'maxChildren' in kwargs:
        model = LocalGP.LocalGPModel(likelihood,kernelClass(ard_num_dims=2),inheritKernel=True,w_gen=w_gen,maxChildren=kwargs['maxChildren'])
    #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
    else:
        model = LocalGP.LocalGPModel(likelihood,kernelClass(ard_num_dims=2),inheritKernel=True,w_gen=w_gen)
    return model

def makeSplittingModel(kernelClass,likelihood,splittingLimit):
    #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
    model = SplittingLocalGP.SplittingLocalGPModel(likelihood,kernelClass(ard_num_dims=2),splittingLimit,inheritKernel=True)
    return model
    
def makeLocalGPModels(kernelClass,likelihood,w_gen,k,**kwargs):
    models = []
    for i in range(k):
        if 'maxChildren' in kwargs:
            models.append(makeLocalModel(kernelClass, likelihood, w_gen, maxChildren=kwargs['maxChildren']))
        else:
            models.append(makeLocalModel(kernelClass, likelihood, w_gen))
    return models

def makeSplittingLocalGPModels(kernelClass,likelihood,splittingLimit,k):
    models = []
    for i in range(k):
        models.append(makeSplittingModel(kernelClass, likelihood, splittingLimit))
    return models

'''
Run a k-fold cross validation of a model. Time is recorded by computing t1-t0, where t1
is the end time of the crossvalidation, and t0 is the start time. To avoid recomputing
the model for earlier data points, we just fit the existing model with 100 more data points,
then add the additional computation time to the existing record. withheldPointsList contains k entries,
each of which is a list containing the points witheld in the previous model. This avoids
recomputing the witheld points each iteration.
'''
def kFoldCrossValidation(modelsList,numSamples,completeRandIndices,predictors,response,modelType,withheldPointsIndices=[]):
    global logger
    # of folds is equal to # of models given
    models = copy.deepcopy(modelsList)
    k = len(models)
    #Take only the new 100 data points to be added
    randIndices = completeRandIndices[numSamples-100:numSamples]
    
    predictions = []
    meanSquaredErrors = []
    elapsedTrainingTimes = []
    memoryUsages = []
    
    for index,model in zip(range(k),models):
        #Choose 100/k data points to withhold. Note k must divide number of data points.
        sliceStart = int(index*100/k)
        sliceEnd = int((index+1)*100/k)
        #These are the indices of the 100 new points being considered
        includedPointsIndices = list(range(randIndices.shape[-1]))
        #These are the points out of 100 which are held out for cross validation
        newlyWithheldPoints = list(map(lambda i: i + numSamples - 100,includedPointsIndices[sliceStart:sliceEnd]))
        del includedPointsIndices[sliceStart:sliceEnd]
        
        t0 = time.time()
        #Train the model with 1/k th of the data withheld
        for ptno,i in zip(range(len(includedPointsIndices)),includedPointsIndices):
            randIndex = randIndices[i]
            
            #Make the input 2D or the model think it is just 2 different inputs
            x_train = predictors[randIndex].unsqueeze(0)
            y_train = response[randIndex].unsqueeze(0)
            
            childCount = len(model.children)
            
            model.update(x_train,y_train)
            
            if(childCount>0 and len(model.children)>childCount):
                model.lastSplit = numSamples - 100 + ptno + 1
                print('split: {0}'.format(model.lastSplit))
                
        t1 = time.time()
        print('Done with fold {0}'.format(index))    
        
        #Add newly withheld points to get OOB MSE
        withheldPointsIndices[index] += newlyWithheldPoints
        
        #withheldPointsIndices = [index for index in range(numSamples) if index not in includedPointsIndices]
        
        #Predict at withheld points for calculating out of bag MSE
        randIndicesWithheld = completeRandIndices[withheldPointsIndices[index]]
        randDataWithheld = predictors[randIndicesWithheld].unsqueeze(0)
        
        #Need to unpack these since we are returning local predictions now
        results = model.predict(randDataWithheld)
        
        #Exact models do not track additional info about how prediction is computed
        if modelType=='exact':
            prediction = results
        else:
            prediction,localPredictions,localWeights,minDists = results[0],results[1],results[2],results[3]
    
        predictions.append(prediction)
        mse = torch.sum(torch.pow(prediction-response[randIndicesWithheld],2),dim=list(range(prediction.dim())))/(numSamples/k)
       
         #If the MSE is very high or nan, log some key info to debug
        if(mse<0 or mse!=mse):
            newTestInds = completeRandIndices[newlyWithheldPoints]
            newTestPoints = predictors[newTestInds].unsqueeze(0)    
            trainingData = []
            kernelHyperParams = []
            covarMatrices = []
            centers = []
            numObsList = []
            fold = k
            lastSplit = model.lastSplit
            
            squaredErrors = torch.pow(prediction-response[randIndicesWithheld],2)
            
            for child in model.children:
                trainingData += child.train_inputs
                kernelHyperParams.append(child.covar_module.lengthscale)
                #covarMatrix = child.covar_module(newTestPoints).evaluate()
                #covarMatrices += covarMatrix
                centers.append(child.center)
                numObsList.append(child.train_inputs[0].size())
                
            logger.log_mse_spike(fullTestPoints=randDataWithheld,
                                 newTestPoints=newTestPoints,
                                 trainingPoints=trainingData,
                                 kernelHyperParams=kernelHyperParams,
                                 covarMatrices=covarMatrices,
                                 centers=centers,
                                 numObs=numSamples,
                                 numObsList=numObsList,
                                 fold=fold,
                                 mse=mse,
                                 squaredErrors=squaredErrors,
                                 lastSplit=lastSplit,
                                 prediction=prediction.detach().squeeze(0).squeeze(-1),
                                 groundTruth=response[randIndices],
                                 localPredictions=localPredictions,
                                 localWeights=localWeights,
                                 minDists=minDists)
        
        meanSquaredErrors.append(mse.detach())
        elapsedTrainingTimes.append(t1-t0)
        memoryUsages.append(MemoryHelper.getMemoryUsage())
    
    return {'mse':meanSquaredErrors,'training_time':elapsedTrainingTimes,'memory_usage':memoryUsages},models,withheldPointsIndices

def resultsToDF(results,modelType,params):
    for key in results:
        mse = results[key]['mse']
        mse = list(map(lambda x: float(x.detach()),mse))
        results[key] = {'mse':mse,'training_time':torch.tensor(results[key]['training_time']),
                        'memory_usage':results[key]['memory_usage'],
                        'replicate':results[key]['replicate']}
    
    paramDict = {'exact':'fantasyUpdate','splitting':'splittingLimit','local':'w_gen'}
    
    df = pd.DataFrame(results).transpose()
    
    df['avg_mse'] = df['mse'].apply(np.mean)
    #Need to cumulatively sum the entries since we only recorded the time to update each model
    df['training_time'] = df['training_time'].cumsum().tolist()
    df['avg_training_time'] = df['training_time'].apply(torch.mean)
    df['avg_training_time'] = df['avg_training_time'].apply(float)
    df['avg_training_time_per_update'] = df['avg_training_time']/df.index
    df['avg_memory_usage'] = df['memory_usage'].apply(np.mean)
    df['model'] = modelType
    
    paramName = paramDict[modelType]
    
    df['params'] = '{0}={1}'.format(paramName,params[paramName])
    df['replicate'] = results[list(results.keys())[0]]['replicate']
    
    return df

#Run a single replicate of the experiment
def runReplicate(replicate, seed, maxSamples, predictors, response, modelType, getModelsFunction):    
    
    #Create the models to be cross validated
    modelsList = getModelsFunction()
    
    for model in modelsList:
        model.lastSplit = -1
    
    #Set the seed for the RNG
    torch.manual_seed(seed)
    
    #Sample some random points
    completeRandIndices = torch.multinomial(torch.ones((predictors.shape[0])).float(),maxSamples,replacement=False)
    withheldPointsIndices = [[] for i in range(len(modelsList))]
    
    numSamplesTensor = torch.linspace(100,maxSamples,maxSamples//100)
    results = {}
    for i in range(numSamplesTensor.shape[-1]):
        numSamples = int(numSamplesTensor[i])
        print('numSamples={0}'.format(100*(i+1)))
        results[numSamples],modelsList,withheldPointsIndices = kFoldCrossValidation(modelsList,
                                                                 numSamples,
                                                                 completeRandIndices,
                                                                 predictors,
                                                                 response,
                                                                 modelType,
                                                                 withheldPointsIndices
                                                                 )
        results[numSamples]['replicate'] = replicate

    return results

'''
Create and run a new experiment for analyzing a GP model. Replicates of the experiment are performed in parallel using the multiprocessing
module.

Arguments:

    modelType: A string giving the type of GP model to test. Must be one of the following: 'splitting','local','exact'

Keyword Arguments:
    
    params: A dict which specifies the model hyperparameters. For a splitting model, this is 'splittingLimit'.
        For a local model, this is 'w_gen'. Exact models require no hyperparameters.
        
    folds: # of folds in the crossvalidation. Must be an int greater than 1. Don't make this too big. Default is 5.
        
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
    
    global logger
    
    logger = GPLogger.ExperimentLogger('{0}-log-{1}.txt'.format(modelType,int(time.time())))
    
    params = kwargs['params']
    kernel = gpytorch.kernels.RBFKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood
    
    #Load icethick dataset
    predictor,response = TestData.icethick(scale=False)
    
    #Set # of folds for cross-validation
    k = kwargs['folds'] if 'folds' in kwargs else 5
    
    #Run k-fold cross validation for num samples ranging from 100-maxSamples
    maxSamples = kwargs['maxSamples'] if 'maxSamples' in kwargs else 2500
    
    #Set # of replications to perform
    replications = kwargs['replications'] if 'replications' in kwargs else 30
    
    #Generate a seed for each replicate
    torch.manual_seed(6942069)
    seeds = list(map(lambda x: int(x),torch.floor(100000*torch.rand(size=(replications,1)))))
    replicates = range(replications)
    
    
    global getModelsFunction
    
    #Create a function to get new models for each step of the experiment
    if modelType=='exact':
        def getModelsFunction(): return makeExactModels(kernel, likelihood, k, inheritKernel=True, fantasyUpdate=params['fantasyUpdate'])
    elif modelType=='local':
        def getModelsFunction(): return makeLocalGPModels(kernel, likelihood, params['w_gen'], k)
    elif modelType=='splitting':
        def getModelsFunction(): return makeSplittingLocalGPModels(kernel, likelihood, params['splittingLimit'], k)
    
    #Multiprocessing hangs for some reason... use loop for now
    '''
    #Create multiprocessing pool for each replicate needed
    pool = mp.Pool(replications)
    '''
    replicateArgsList = [(replicate, seed, maxSamples, predictor, response, modelType, getModelsFunction) for replicate,seed in zip(replicates,seeds)]
    
    '''
    results = pool.starmap(runReplicate,replicateArgsList)
    
    pool.close()
    '''
    
    results = [runReplicate(*args) for args in replicateArgsList]
    
    #Convert results dicts into dataframes, then merge
    experimentDF = pd.concat(list(map(lambda resultDict:resultsToDF(resultDict, modelType, params),results)))
    
    return experimentDF