# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:46:50 2020

@author: pnter
"""


import rBCM
import TestData
import numpy as np
import torch
import pandas as pd


#params = [256,64,16,4]
params = [32]
numReps = 1

rmseDict = {}
stdDict = {}

torch.manual_seed(41064)
seeds = torch.ceil(torch.rand((numReps,1))*100000).long()
'''
for numChildren in params:
    
    rmseArray = np.zeros((10,1))
    for rep in range(numReps):
        
        seed = seeds[rep]
        
        #Load data and convert to numpy
        predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.powergen(seed,scale=False)
        predictorsTrain,responseTrain,predictorsTest,responseTest = [x.double().numpy() for x in [predictorsTrain,responseTrain,predictorsTest,responseTest]]
        
        #Create rBCM model
        model = rBCM.rBCM(predictorsTrain,responseTrain,
                          cov_type='covSEard',
                          pool=None,
                          profile=[(numChildren,'random','rep1')])
        
        model.train()
        preds = model.predict(predictorsTest)
        rmse = np.sqrt(np.mean((preds-responseTest)**2))
        rmseArray[rep] = rmse
        print('Done with {0} children!'.format(numChildren))
    
    rmseDict[numChildren] = np.mean(rmseArray)
    stdDict[numChildren] = np.std(rmseArray)
    
df = pd.DataFrame()
df['numChildren'] = params
df['mean_rmse'] = list(rmseDict.values())
df['std_rmse'] = list(stdDict.values())

df.to_csv('rBCM_powergen_results.csv')
'''