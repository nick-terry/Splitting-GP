# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:37:52 2020

@author: pnter
"""


import LocalGP_Kfold_Crossvalidation_TestData
from LocalGP_Kfold_Crossvalidation_TestData import makeExactModels,makeLocalGPModels,makeSplittingLocalGPModels
import GPLogger
import time
import gpytorch
import TestData
import torch

modelType = 'splitting'
maxSamples = 100
params = {'splittingLimit':300}
replications = 1
folds=5

kwargs = {'modelType':modelType,'maxSamples':maxSamples,'params':params,'replications':replications,'folds':folds}

global logger
   
logger = GPLogger.ExperimentLogger('{0}-log-{1}.txt'.format(modelType,int(time.time())))

params = kwargs['params']
kernel = gpytorch.kernels.RBFKernel
likelihood = gpytorch.likelihoods.GaussianLikelihood

#Load icethick dataset
predictor,response = TestData.icethick()

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
    
replicateArgsList = [(replicate, seed, maxSamples, predictor, response, modelType, getModelsFunction) for replicate,seed in zip(replicates,seeds)]
    
#results = LocalGP_Kfold_Crossvalidation_TestData.runReplicate(0, seeds[0], maxSamples, predictor, response, modelType, getModelsFunction)

def start():
    if __name__ == "__main__":
        results = LocalGP_Kfold_Crossvalidation_TestData.runCrossvalidationExperiment(**kwargs)
    return results
