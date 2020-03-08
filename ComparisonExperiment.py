# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:11:52 2020

@author: pnter
"""

import LocalGP_Kfold_Crossvalidation
import pandas as pd
import multiprocessing as mp
'''
modelTypes = ['splitting','exact','local']
maxSamples = 2500
paramsList = [{'splittingLimit':500},None,{'w_gen':.5}]
replications = 30
folds=5
'''

modelTypes = ['splitting']
maxSamples = 2500
paramsList = [{'splittingLimit':500}]
replications = 30
folds=5

def runExperimentSingleCore():
    resultsList = []
    for modelType,params in zip(modelTypes,paramsList):
        results = LocalGP_Kfold_Crossvalidation.runCrossvalidationExperiment(modelType, 
                                                                         params=params,
                                                                         replications=replications,
                                                                         folds=folds,
                                                                         maxSamples=maxSamples)
        resultsList.append(results)
    
    return pd.concat(resultsList)
        
def runExperimentMultiCore():
    argsList = [(modelType,params,replications,folds,maxSamples) for modelType,params in zip(modelTypes,paramsList)]
    #dont attempt to create more workers than cores, or more than the necessary # of jobs
    if __name__ == '__main__':
        pool = mp.Pool(min(mp.cpu_count(),len(argsList)))    
        results = pool.apply(LocalGP_Kfold_Crossvalidation.runCrossvalidationExperiment,args=argsList)
        pool.close()
        print(results.get())
        results = pd.concat(results)
    
        return results

results = runExperimentSingleCore()
results.to_csv('experiment-results-{0}.csv'.format(modelTypes[0]))