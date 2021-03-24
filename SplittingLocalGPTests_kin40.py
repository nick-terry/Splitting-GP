# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import time
import LocalGP
import SplittingLocalGP,RegularGP#,RBCM
import torch
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from itertools import product
from math import inf
import TestData
import pandas as pd

def getIcethick():
    predictor,response,predTest,respTest = TestData.icethick(scale=False)
    return predictor,response,predTest,respTest

def getKin40():
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.kin40()
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double()

def getFires():
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.forestfire()
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double()

def getPowergen(seed):
    predictorsTrain,responseTrain,predictorsTest,responseTest = TestData.powergen(seed)
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double()

def getSine():
    predictorsTrain,responseTrain,predictorsTest,responseTest,predictor,response = TestData.onedimsine()
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double(),predictor.double(),response.double()

def getStep():
    predictorsTrain,responseTrain,predictorsTest,responseTest,predictor,response = TestData.onedimstep()
    return predictorsTrain.double(),responseTrain.double(),predictorsTest.double(),responseTest.double(),predictor.double(),response.double()

'''
predictorsTrain,responseTrain,predictorsTest,responseTest = getFires()
#log10 transform of response
responseTrain = torch.log10(responseTrain+1)
'''
'''
predictorsTrain,responseTrain,predictorsTest,responseTest,predictor,response = getSine()
'''

#predictorsTrain,responseTrain,predictorsTest,responseTest = getPowergen()


def makeModel(kernelClass,likelihood,M,splittingLimit,inheritLikelihood):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = SplittingLocalGP.SplittingLocalGPModel(likelihood,gpytorch.kernels.ScaleKernel(kernelClass(ard_num_dims=4)),
                                                       splittingLimit=splittingLimit,inheritKernel=True,
                                                       inheritLikelihood=inheritLikelihood,
                                                       M=M,
                                                       mean=gpytorch.means.ConstantMean)
        return model

def makeLocalModel(kernelClass,likelihood,M,w_gen):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = LocalGP.LocalGPModel(likelihood,gpytorch.kernels.ScaleKernel(kernelClass(ard_num_dims=4)),
                                                       w_gen=w_gen,inheritKernel=True,
                                                       M=M,
                                                       mean=gpytorch.means.ConstantMean)
        return model

def makeRegularModel(kernelClass,likelihood):
        model = RegularGP.RegularGPModel(likelihood,kernelClass(ard_num_dims=8))
        return model
   
def makeRBCMModel(kernelClass,likelihood,k):
        #Note: ard_num_dims=2 permits each input dimension to have a distinct hyperparameter
        model = RBCM.RobustBayesCommitteeMachine(likelihood,kernelClass(ard_num_dims=8),
                                                       inheritKernel=True,
                                                       numChildren=k,
                                                       mean=gpytorch.means.ZeroMean)
        return model
   
def makeModels(kernelClass,likelihood,w_gen,k):
    models = []
    for i in range(k):
        models.append(makeModel(kernelClass, likelihood, w_gen))
    return models

def evalModel(M=None,splittingLimit=500,inheritLikelihood=True,mtype='splitting',w_gen=.5,k=10):
    #Set RNG seed
    torch.manual_seed(42069)
        
    kernel = gpytorch.kernels.RBFKernel

    likelihood = gpytorch.likelihoods.GaussianLikelihood
    
    
    if mtype=='splitting':
        model = makeModel(kernel,likelihood,M,splittingLimit,inheritLikelihood)
    elif mtype=='local':
        model = makeLocalModel(kernel,likelihood,M,w_gen=w_gen)
    elif mtype=='rbcm':
        model = makeRBCMModel(kernel,likelihood,k)
    else:
        model = makeRegularModel(kernel,likelihood)
        
    t0 = time.time()
    j = 0
    if mtype=='splitting':
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
    
    elif mtype=='local':
        for index in range(int(predictorsTrain.shape[0])):
            
            x_train = predictorsTrain[index].unsqueeze(0)
            y_train = responseTrain[index].unsqueeze(0)
            model.update(x_train,y_train)
            j += 1
            
    elif mtype=='rbcm':
        for index in range(int(predictorsTrain.shape[0]))[::100]:
            #We don't want to overshoot the number of obs...
            upperIndex = min(index+100,int(predictorsTrain.shape[0]))
            x_train = predictorsTrain[index:upperIndex]
            y_train = responseTrain[index:upperIndex]
            model.update(x_train,y_train)
            print(j)
            j += 1
    
    else:
        model.update(predictorsTrain,responseTrain)

    t1 = time.time()
    print('Done training')
    
    '''
    #Predict over the whole grid for plotting
    totalPreds = model.predict(predictorsTest,individualPredictions=False)
    prediction = totalPreds[0].detach()
    '''
    
    return model,t1-t0

def evalgptModel():
    
    train_x = predictorsTrain.double()
    train_y = responseTrain.double()
    class ExactGPModel(gpytorch.models.ExactGP):
        
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = ExactGPModel(train_x, train_y, likelihood)
    likelihood = model.likelihood
    model.double()
    likelihood.double()
    model.train()
    likelihood.train()
    model.train_inputs = (torch.cat([model.train_inputs[0], predictorsTrain]),)
    model.train_targets = torch.cat([model.train_targets, responseTrain])
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll.double()
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.train_inputs[0])
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model

#Splitting
#kin40

'''
paramsList = [{'M':1,'splittingLimit':70,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':156,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':625,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':2500,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':5000,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':9999,'inheritLikelihood':False,'mtype':'splitting'}]
'''

'''
paramsList = [{'M':1,'splittingLimit':70,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':70,'inheritLikelihood':False,'mtype':'splitting'}]
'''

#powergen
paramsList = [{'M':None,'splittingLimit':625,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':300,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':150,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':75,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':30,'inheritLikelihood':False,'mtype':'splitting'}]
'''
paramsList = [{'M':None,'splittingLimit':5000,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':2500,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':1250,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':625,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':300,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':150,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':None,'splittingLimit':75,'inheritLikelihood':False,'mtype':'splitting'}]
'''
'''
paramsList = [{'M':None,'splittingLimit':500,'inheritLikelihood':True,'mtype':'splitting'}]

'''
'''
              {'M':1,'splittingLimit':156,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':625,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':2500,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':5000,'inheritLikelihood':False,'mtype':'splitting'},
              {'M':1,'splittingLimit':9999,'inheritLikelihood':False,'mtype':'splitting'}]
'''

#Fires

'''
paramsList = [{'M':1,'splittingLimit':100,'inheritLikelihood':True,'mtype':'splitting'},
              {'M':1,'splittingLimit':75,'inheritLikelihood':True,'mtype':'splitting'},
              {'M':1,'splittingLimit':50,'inheritLikelihood':True,'mtype':'splitting'},
              {'M':1,'splittingLimit':25,'inheritLikelihood':True,'mtype':'splitting'},
              {'M':1,'splittingLimit':10,'inheritLikelihood':True,'mtype':'splitting'}]
'''

#RBCM

'''
paramsList = [{'M':1,'k':10,'inheritLikelihood':True,'mtype':'rbcm'},
              {'M':1,'k':128,'inheritLikelihood':True,'mtype':'rbcm'},
              {'M':1,'k':64,'inheritLikelihood':True,'mtype':'rbcm'},
              {'M':1,'k':32,'inheritLikelihood':True,'mtype':'rbcm'},
              {'M':1,'k':16,'inheritLikelihood':True,'mtype':'rbcm'},
              {'M':1,'k':1,'inheritLikelihood':True,'mtype':'rbcm'}]
'''

#Local

#best so far for powergen is 1e-44
'''
paramsList = [{'M':None,'w_gen':1*10**-30,'inheritLikelihood':True,'mtype':'local'}]
'''

'''
t0 = time.time()
model = evalgptModel()
t1 = time.time()
model.eval()
model.likelihood.eval()

preds = model.likelihood(model(predictorsTest)).mean
rmse = torch.sqrt(torch.mean((preds-responseTest)**2))
mad = torch.mean(torch.abs(preds-responseTest))
df = pd.DataFrame()
df['params'] = ['full gp']
df['time'] = [t1-t0]
df['rmse'] = [rmse.detach().numpy()]
df['mad'] = [mad.detach().numpy()]
df.to_csv('powergen_results_gp.csv')
'''

#paramsList = [{'M':1,'w_gen':.0001,'inheritLikelihood':True,'mtype':'local'}]

#paramsList = [{'M':None,'splittingLimit':75,'inheritLikelihood':True,'mtype':'splitting'}]

#Number of replications to perform. Only necessary to do >1 if randomized
numReps = 10

#Create seeds
#torch.manual_seed(41064)
#seeds = torch.ceil(torch.rand((numReps,1))*100000).long()

torch.manual_seed(10101)
np.random.seed(10101)
seeds = [41065,10342,98891,36783,11102,34522,98991,98990,76766,27726]

madArr = torch.zeros((len(paramsList)*numReps,1))
resultsArr = torch.zeros((len(paramsList)*numReps,1))
timeArr = torch.zeros((len(paramsList)*numReps,1))
repArr = torch.arange(numReps).repeat_interleave(len(paramsList))

#predictorsTrain,responseTrain,predictorsTest,responseTest,predictor,response = getSine()
'''
fig = plt.figure()
ax = fig.subplots(2)
'''
#may need to update the max iterations here to prevent numerical issues with rbcm
with gpytorch.settings.max_cg_iterations(20000):
    for i in range(len(paramsList)):
        for rep in range(numReps):
            seed = seeds[rep]
            predictorsTrain,responseTrain,predictorsTest,responseTest = getPowergen(seed)
            
            t0 = time.time()
            model,deltaT = evalModel(**paramsList[i])
            if paramsList[i]['mtype']=='rbcm':    
                preds,variances = model.predict(predictorsTest)
            else:
                preds = model.predict(predictorsTest)
            t1 = time.time()
            rmse = torch.sqrt(torch.mean((preds-responseTest)**2))
            print(rmse)
            mad = torch.mean(torch.abs(preds-responseTest))
            resultsArr[i+rep*len(paramsList)] = rmse
            timeArr[i+rep*len(paramsList)] = t1-t0
            madArr[i+rep*len(paramsList)] = mad

#create dataframe for storing experiment results

df = pd.DataFrame()

concatParams = []
for rep in range(numReps):
    concatParams += paramsList

df['params'] = concatParams
df['time'] = timeArr.detach().numpy()
df['rmse'] = resultsArr.detach().numpy()
df['mad'] = madArr.detach().numpy()
df['replication'] = repArr.detach().numpy()

df.to_csv('powergen_results_splitting_10reps.csv')


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