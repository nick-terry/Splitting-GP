# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import time
import LocalGP
import SplittingLocalGP,RegularGP
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from math import inf
import TestData
import pandas as pd
import sklearn as skl


def getSKLearnModel():
    model = skl.gaussian_process.GaussianProcessRegressor()
    return model

def getIcethick():
    predictor,response = TestData.icethick(scale=False)
    return predictor,response

def getKin40():
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.kin40()
    return predictorsTrain,responseTrain,predictorsTest,responseTest

predictorsTrain,responseTrain,predictorsTest,responseTest = getKin40()

def makeModel(kernelClass,likelihood,M,splittingLimit,inheritLikelihood):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = SplittingLocalGP.SplittingLocalGPModel(likelihood,kernelClass(ard_num_dims=8),
                                                       splittingLimit=splittingLimit,inheritKernel=True,
                                                       inheritLikelihood=True,
                                                       M=M,
                                                       mean=gpytorch.means.ZeroMean)
        return model

def makeRegularModel(kernelClass,likelihood):
        model = RegularGP.RegularGPModel(likelihood,kernelClass(ard_num_dims=8))
        return model
    
def makeModels(kernelClass,likelihood,w_gen,k):
    models = []
    for i in range(k):
        models.append(makeModel(kernelClass, likelihood, w_gen))
    return models

def evalModel(M=None,splittingLimit=500,inheritLikelihood=True,splitting=True):
    #Set RNG seed
    torch.manual_seed(42069)
        
    kernel = gpytorch.kernels.RBFKernel

    likelihood = gpytorch.likelihoods.GaussianLikelihood
    
    
    if splitting:
        model = makeModel(kernel,likelihood,M,splittingLimit,inheritLikelihood)
    else:
        model = makeRegularModel(kernel,likelihood)
        
    t0 = time.time()
    j = 0
    if splitting:
        for index in range(predictorsTrain.shape[0]):
            x_train = predictorsTrain[index].unsqueeze(0)
            y_train = responseTrain[index].unsqueeze(0)
            model.update(x_train,y_train)
            print(j)
            j += 1
    else:
        model.update(predictorsTrain,responseTrain)

    t1 = time.time()
    print('Done training')
    
    #Predict over the whole grid for plotting
    totalPreds = model.predict(predictorsTest,individualPredictions=False)
    prediction = totalPreds[0].detach()
    
    
    rmse = torch.sqrt(torch.mean((prediction.squeeze(-1)-responseTest)**2))
    print(rmse)
    return rmse


paramsList = [{'M':None,'splittingLimit':500,'inheritLikelihood':True,'splitting':True}]

resultsArr = torch.zeros((len(paramsList),1))

for i in range(len(paramsList)):
    resultsArr[i] = evalModel(**paramsList[i])    

df = pd.DataFrame()
df['params'] = paramsList
df['rmse'] = resultsArr.numpy()
df.to_csv('kin40_results_splitting.csv')

'''
#Define a common scale for color mapping for contour plots
maxAbsVal = torch.max(torch.abs(response))
levels = np.linspace(-maxAbsVal,maxAbsVal,30)

#Plot true function
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,5))
contours = axes[0].scatter(predictor[:,0].detach(),predictor[:,1].detach(),c=response)

#Plot GP regression approximation
axes[1].scatter(predictor[:,0].detach(),predictor[:,1].detach(),c=prediction.squeeze(-1))

childrenCenters = model.getCenters().squeeze(1)
axes[1].scatter(childrenCenters[:,0].detach(),childrenCenters[:,1].detach(),c='orange',s=24,edgecolors='white')
'''
#Show the points which were sampled to construct the GP model
'''
sampledPoints  = predictor[randIndices]
axes[1].scatter(sampledPoints[:,0].detach(),sampledPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
'''