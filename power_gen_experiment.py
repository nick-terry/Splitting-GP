# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import time
import SplittingLocalGP
import torch
import gpytorch
import numpy as np
import TestData
import pandas as pd

def getKin40():
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.kin40()
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double()

def getPowergen(seed):
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.powergen(seed)
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double()

def makeModel(kernelClass,likelihood,M,splittingLimit,inheritLikelihood):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = SplittingLocalGP.SplittingLocalGPModel(likelihood,gpytorch.kernels.ScaleKernel(kernelClass(ard_num_dims=4)),
                                                       splittingLimit=splittingLimit,inheritKernel=True,
                                                       inheritLikelihood=inheritLikelihood,
                                                       M=M,
                                                       mean=gpytorch.means.ConstantMean)
        return model

def evalModel(M=None,splittingLimit=500,inheritLikelihood=True,mtype='splitting'):
    #Set RNG seed
    torch.manual_seed(42069)
        
    kernel = gpytorch.kernels.RBFKernel

    likelihood = gpytorch.likelihoods.GaussianLikelihood
    
    
    model = makeModel(kernel,likelihood,M,splittingLimit,inheritLikelihood)

        
    t0 = time.time()
    j = 0
    for index in range(int(predictorsTrain.shape[0]))[::splittingLimit]:
        #We don't want to overshoot the number of obs...
        upperIndex = min(index+splittingLimit,int(predictorsTrain.shape[0]))
        x_train = predictorsTrain[index:upperIndex]
        y_train = responseTrain[index:upperIndex]
        
        #Need to unsqueeze if the data is 1d
        if x_train.dim()==1:
            x_train = x_train.unsqueeze(1)
            y_train = y_train
        
        
        model.update(x_train,y_train)
        print(j)
        j += 1

    t1 = time.time()
    print('Done training')
    
    return model,t1-t0


#Define the parameters we will use for each model tested.
paramsList = [{'M':None,'splittingLimit':625,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':300,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':150,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':75,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':30,'inheritLikelihood':False,'mtype':'splitting'}]

#Number of replications to perform. Only necessary to do >1 if randomized
numReps = 10

#Fix the random seeds for using CRN with rBCM experiment
torch.manual_seed(10101)
np.random.seed(10101)
seeds = [41065,10342,98891,36783,11102,34522,98991,98990,76766,27726]

#Create arrays to store experimental results
madArr = torch.zeros((len(paramsList)*numReps,1))
resultsArr = torch.zeros((len(paramsList)*numReps,1))
timeArr = torch.zeros((len(paramsList)*numReps,1))
repArr = torch.arange(numReps).repeat_interleave(len(paramsList))

#Up this setting to prevent potential numerical issues if CG hasn't converged in <2000 iterations
with gpytorch.settings.max_cg_iterations(2000):
    for i in range(len(paramsList)):
        for rep in range(numReps):
            
            #Take the seed for current rep, an retrieve the powergen dataset. Seed is used to create random 80/20 train/test split
            seed = seeds[rep]
            predictorsTrain,responseTrain,predictorsTest,responseTest = getPowergen(seed)
            
            #Record time for benchmarking
            t0 = time.time()

            #Construct and fit model
            model,deltaT = evalModel(**paramsList[i])
            
            #Create predictions using the test data
            preds = model.predict(predictorsTest)

            #Record the stopping time
            t1 = time.time()
            
            #Compute root mean squared error, mean absolute deviation
            rmse = torch.sqrt(torch.mean((preds-responseTest)**2))
            mad = torch.mean(torch.abs(preds-responseTest))
            
            resultsArr[i+rep*len(paramsList)] = rmse
            timeArr[i+rep*len(paramsList)] = t1-t0
            madArr[i+rep*len(paramsList)] = mad

#Create dataframe for storing experiment results
df = pd.DataFrame()

concatParams = []
for rep in range(numReps):
    concatParams += paramsList

df['params'] = concatParams
df['time'] = timeArr.detach().numpy()
df['rmse'] = resultsArr.detach().numpy()
df['mad'] = madArr.detach().numpy()
df['replication'] = repArr.detach().numpy()

#Write results to CSV
df.to_csv('powergen_results_splitting_10reps.csv')