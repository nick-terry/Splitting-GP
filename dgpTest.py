# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:46:50 2020

@author: pnter
"""


import RBCM
import TestData
import numpy as np
import torch
import pandas as pd 
import time


'''
The original experiment was to run with these parameter values, but 
we found that E=254 resulting in numerical issues (NAN values for loglikelihood).
'''
#params = [12,25,50,102,254]

params = [12,25,50,102]
numReps = 10

'''
torch.manual_seed(41065)
seeds = torch.ceil(torch.rand((numReps,1))*1000).long()
'''
#seeds = [41065,10342,98891,36788,11102]
np.random.seed(10101)
seeds = [41065,10342,98891,36783,11102,34522,98991,98990,76766,27726] #27726 

rmseArray = np.zeros((len(params)*numReps,1))
timeArray = np.zeros((len(params)*numReps,1))
repArr = torch.arange(numReps).repeat_interleave(len(params))

for i in range(len(params)):
        
        for rep in range(numReps):
            
            seed = seeds[rep]
            
            numChildren = params[i]
            
            #Load data and convert to numpy
            predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.powergen(seed)
            predictorsTrain,responseTrain,predictorsTest,responseTest = [x.double().numpy() for x in [predictorsTrain,responseTrain,predictorsTest,responseTest]]
            
            #Create rBCM model
            model = RBCM.rBCM(predictorsTrain,responseTrain,
                              cov_type='covSEard',
                              pool=None,
                              profile=[(numChildren,'random','rep1')])
            
            t0 = time.time()
            model.train()
            preds = model.predict(predictorsTest)
            t1 = time.time()
        
            rmse = np.sqrt(np.mean((preds-responseTest)**2))
            rmseArray[i+rep*len(params)] = rmse
            timeArray[i+rep*len(params)] = t1-t0
        
            print('Done with {0} children, replicate {1}!'.format(numChildren,rep))

concatParams = []
for rep in range(numReps):
    concatParams += params

df = pd.DataFrame()
df['numChildren'] = concatParams
df['replicate'] = repArr
df['rmse'] = rmseArray
df['time'] = timeArray

df.to_csv('rBCM_powergen_results_ppe_newformat.csv')
