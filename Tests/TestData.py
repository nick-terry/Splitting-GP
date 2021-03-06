# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:32:42 2020

@author: pnter
"""

import torch
import pandas as pd
import UtilityFunctions
from pathlib import Path
import os.path
from math import ceil,pi
import numpy as np

path = Path().absolute()

def icethick(full=False,scale=False,project=False):
    df = pd.read_csv('ICETHK.csv')
    if full:
        return df
    
    else:
        df = df[['Longitude W','Latitude S','Thickness (m)']]
        
        #Transform into PyTorch tensors of predictors and responses
        x = torch.tensor(df[['Longitude W','Latitude S']].to_numpy())
        y = torch.tensor(df['Thickness (m)'].to_numpy())
        
        #Get every 3rd element for now to test on smaller scale
        '''
        x = x[::3,:].float()
        y = y[::3].float()
        '''
        
        if project:
            #center = torch.mean(x,dim=0)
            southPole = torch.tensor([-90,0]).float()
            projectFn = UtilityFunctions.getGnomomic(southPole)
            x = projectFn(x)
            
        if scale:
            x[:,0] = x[:,0]/torch.max(torch.abs(x[:,0]))
            x[:,1] = x[:,1]/torch.max(torch.abs(x[:,1]))
            y = y/torch.max(y) #Scale down for regression
        
        indices = torch.multinomial(torch.ones((x.shape[0])).float(),ceil(x.shape[0]*.8),replacement=False)
        
        predictorTrain,responseTrain = x[indices],y[indices]
        complementMask = torch.ones(x.shape[0]).bool()
        complementMask[indices] = False
        predictorTest,responseTest = x[complementMask],y[complementMask]
    
        return predictorTrain,responseTrain,predictorTest,responseTest

def kin40():
    trainInputs = torch.tensor(pd.read_fwf('kin40k_train_inputs.txt',header=None).to_numpy()).float()
    testInputs = torch.tensor(pd.read_fwf('kin40k_test_inputs.txt',header=None).to_numpy()).float()
    trainLabels = torch.tensor(pd.read_fwf('kin40k_train_labels.txt',header=None).to_numpy()).squeeze().float()
    testLabels = torch.tensor(pd.read_fwf('kin40k_test_labels.txt',header=None).to_numpy()).squeeze().float()
    
    return trainInputs,trainLabels,testInputs,testLabels

def forestfire():
    
    df = pd.read_csv('forestfires.csv')
    
    predictor = torch.tensor(df[['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain']].to_numpy())
    response = torch.tensor(df['area'].to_numpy())
    
    torch.manual_seed(41064)
    
    indices = torch.multinomial(torch.ones((predictor.shape[0])).float(),ceil(predictor.shape[0]*.9),replacement=False)
    
    predictorTrain,responseTrain = predictor[indices],response[indices]
    complementMask = torch.ones(predictor.shape[0]).bool()
    complementMask[indices] = False
    predictorTest,responseTest = predictor[complementMask],response[complementMask]
    
    return predictorTrain,responseTrain,predictorTest,responseTest

#We want to be able to pass a seed here so we can use CRNG
def powergen(seed=41064,scale=False):
    
    df = pd.read_csv('power_gen.csv')
    
    predictor = torch.tensor(df[['temperature','vaccuum','ambient_pressure','relative_humidity']].to_numpy())
    if scale:
        pred_mean = torch.mean(predictor,dim=0)
        predictor -= pred_mean
        pred_max_abs,indices = torch.max(torch.abs(predictor),dim=0)
        predictor  /= pred_max_abs
        
    response = df['power_output'].to_numpy()
    
    #There are some duplicate predictors. Get unique values.
    predictor,indices = np.unique(predictor,return_index=True,axis=0)
    response = response[indices]
    
    predictor = torch.tensor(predictor)
    response = torch.tensor(response)   
    
    torch.manual_seed(seed)
    
    indices = torch.multinomial(torch.ones((predictor.shape[0])).float(),ceil(predictor.shape[0]*.8),replacement=False)
    
    predictorTrain,responseTrain = predictor[indices],response[indices]
    complementMask = torch.ones(predictor.shape[0]).bool()
    complementMask[indices] = False
    predictorTest,responseTest = predictor[complementMask],response[complementMask]
    
    return predictorTrain,responseTrain,predictorTest,responseTest

def onedimsine():
    
    predictor = torch.linspace(0,20,steps=1000)
    response = torch.sin(predictor)+2*torch.cos(predictor+pi/4)+.2*predictor-.1*predictor**2
    response[:response.shape[0]//2] += torch.randn(response.shape[0]//2)
    response[response.shape[0]//2:] += torch.randn(response.shape[0]//2)*3
    torch.manual_seed(41064)
    
    indices = torch.multinomial(torch.ones((predictor.shape[0])).float(),ceil(predictor.shape[0]*.2),replacement=True)
    
    predictorTrain,responseTrain = predictor[indices],response[indices]
    complementMask = torch.ones(predictor.shape[0]).bool()
    complementMask[indices] = False
    predictorTest,responseTest = predictor[complementMask],response[complementMask]
    
    return predictorTrain,responseTrain,predictorTest,responseTest,predictor,response

def onedimstep():
    
    predictor = torch.linspace(0,20,steps=1000)
    response = torch.zeros(predictor.shape)
    repets = 3
    for i in range(repets):
        response[response.shape[0]*2*i//(2*repets):response.shape[0]*(2*i+1)//(2*repets)] = 1
    
    torch.manual_seed(41064)
    
    indices = torch.multinomial(torch.ones((predictor.shape[0])).float(),ceil(predictor.shape[0]*.5),replacement=True)
    
    predictorTrain,responseTrain = predictor[indices],response[indices]
    complementMask = torch.ones(predictor.shape[0]).bool()
    complementMask[indices] = False
    predictorTest,responseTest = predictor[complementMask],response[complementMask]
    
    return predictorTrain,responseTrain,predictorTest,responseTest,predictor,response

df = powergen()