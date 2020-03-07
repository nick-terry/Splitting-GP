# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:11:52 2020

@author: pnter
"""

import LocalGP_Kfold_Crossvalidation
import pandas as pd
import multiprocessing as mp

#pool = mp.Pool(mp.cpu_count())

modelTypes = ['splitting','exact','local']
maxSamples = 100
paramsList = [{'splittingLimit':500},None,{'w_gen':.5}]
replications = 2
folds=2

combinedResults = None

for modelType,params in zip(modelTypes,paramsList):
    results = LocalGP_Kfold_Crossvalidation.runCrossvalidationExperiment(modelType, 
                                                                     params=params,
                                                                     replications=replications,
                                                                     folds=folds,
                                                                     maxSamples=maxSamples)
    if combinedResults is None:
        combinedResults = results
    else:
        combinedResults = pd.concat([combinedResults,results])
