# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:11:52 2020

@author: pnter
"""

import LocalGP_Kfold_Crossvalidation_TestData
import pandas as pd
import multiprocessing as mp
import logging

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)
'''
modelTypes = ['splitting','exact','local']
maxSamples = 2500
paramsList = [{'splittingLimit':500},None,{'w_gen':.5}]
replications = 30
folds=5
'''

modelTypes = ['splitting']
maxSamples = 100
paramsList = [{'splittingLimit':300}]
replications = 1
folds=5

#Unpacks the args and kwargs to run the experiment
def runExperimentWithKwargs(args):
    return LocalGP_Kfold_Crossvalidation_TestData.runCrossvalidationExperiment(args[0],**args[1])

'''
Run a sequence of experiments for different model types/parameters
'''
def runExperimentSequential():
    resultsList = []
    for modelType,params in zip(modelTypes,paramsList):
        results = LocalGP_Kfold_Crossvalidation_TestData.runCrossvalidationExperiment(modelType, 
                                                                         params=params,
                                                                         replications=replications,
                                                                         folds=folds,
                                                                         maxSamples=maxSamples)
        resultsList.append(results)
    
    return pd.concat(resultsList)
        
def runExperimentMultiCore():
    argsList = [((modelType,
                 {'params':params,'replications':replications,'folds':folds,'maxSamples':maxSamples}),)
                for modelType,params in zip(modelTypes,paramsList)]
    #dont attempt to create more workers than cores, or more than the necessary # of jobs
    if __name__ == '__main__':
        pool = mp.Pool(len(argsList))    
        
        results = pool.starmap(runExperimentWithKwargs,argsList)
        
        pool.close()
        results = pd.concat(results)
    
        return results

results = runExperimentSequential()
results.to_csv('experiment-results-splitting-testdata.csv')


'''
results = runExperimentSingleCore()
results.to_csv('experiment-results-{0}.csv'.format(modelTypes[0]))
'''