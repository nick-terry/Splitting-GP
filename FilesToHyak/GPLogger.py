# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:55:00 2020

@author: pnter
"""

'''
This module provides hooks for logging some recurring events during GP experiments
'''
import logging
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

torch.set_printoptions(profile="full")

class ExperimentLogger:
    
    def __init__(self,logfilename='experiment-log.txt',loggingLevel=logging.DEBUG):
        self.logfilename = logfilename
        self.loggingLevel = loggingLevel
        
        logging.basicConfig(filename=self.logfilename,level=self.loggingLevel)
        
    def log_mse_spike(self,fullTestPoints,newTestPoints,trainingPoints,kernelHyperParams,covarMatrices,centers,numObs,numObsList,fold,mse,squaredErrors,lastSplit,prediction,groundTruth,localPredictions,localWeights):
        
        fullTestPoints = fullTestPoints.squeeze(0)
        newTestPoints = newTestPoints.squeeze(0)
        trainingPoints = torch.cat(trainingPoints)
        
        logging.debug('MSE Spike: {0}'.format(mse))
        logging.debug('Num Obs: {0}'.format(numObs))
        logging.debug('Fold: {0}'.format(fold))
        logging.debug('Last Split: {0}'.format(lastSplit))
        logging.debug('Num Obs per Child:')
        for i in range(len(numObsList)):
            logging.debug('Child {0}: {1} obs'.format(i,numObsList[i]))
        logging.debug('Child Centers:')
        for i in range(len(centers)):
            logging.debug('Child {0}: {1}'.format(i,centers[i]))
        logging.debug('Kernel Lengthscales:\n{0}'.format(kernelHyperParams))
        logging.debug('Covar Matrices:')
        for matrix in covarMatrices:
            logging.debug(str(matrix))
        logging.debug('Full Training Data:\n{0}'.format(trainingPoints))
        logging.debug('Full Test Data:\n{0}'.format(fullTestPoints))
        logging.debug('New Test Data:\n{0}'.format(newTestPoints))
        
        logging.debug('Squared Errors: {0}'.format(squaredErrors))
        logging.debug('Local Predictions and Weights:')
        for i in range(len(localPredictions)):
            logging.debug('Child {0} Pred: {1}'.format(i,localPredictions[i]))
            logging.debug('Child {0} Weight: {1}'.format(i,localWeights[i]))
        
        centers = torch.stack(centers)
        
        self.make_data_plot(trainingPoints,fullTestPoints,newTestPoints,centers,prediction,groundTruth)
        
    def make_data_plot(self,trainingPoints,fullTestPoints,newTestPoints,centers,prediction,groundTruth):
        t = int(time.time())
        gridDims = 100
        scale = 5
        x,y = torch.meshgrid([torch.linspace(-scale,scale,gridDims), torch.linspace(-scale,scale,gridDims)])
        xyGrid = torch.stack([x,y],dim=2).float()
        
        fig,axes = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(12,5))
            
        #Show the points which were sampled to construct the GP model
        axes.scatter(trainingPoints[:,0].detach(),trainingPoints[:,1].detach(),c='blue',s=8,zorder=2)
        axes.scatter(fullTestPoints[:,0].detach(),fullTestPoints[:,1].detach(),c='orange',s=8,zorder=4)
        axes.scatter(newTestPoints[:,0].detach(),newTestPoints[:,1].detach(),c='red',s=8,zorder=6)
        if centers.dim()==1:
            centers = centers.unsqueeze(0)
        axes.scatter(centers[:,0].detach(),centers[:,1].detach(),c='green',s=24,zorder=8)
        plt.savefig('error-data-fig-{0}.png'.format(t))
        
        #Make a plot showing the prediction and ground truth side-by-side    
        fig2,axes2 = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(12,5))
        numPointsIndexer = range(prediction.shape[-1])
        
        axes2.scatter(numPointsIndexer, prediction)
        axes2.scatter(numPointsIndexer, groundTruth)
        yPairs = zip(prediction, groundTruth)
        
        #plt.plot((numPointsIndexer,numPointsIndexer),([i for (i,j) in yPairs], [j for (i,j) in yPairs]),c='black')
            
        axes2.legend(['Predictions','Ground Truth'])
        plt.savefig('error-prediction-fig-{0}.png'.format(t))
        
        plt.close(fig)
        plt.close(fig2)
        
        
        
def plot(trainingPoints,testPoints):
    gridDims = 50
    x,y = torch.meshgrid([torch.linspace(-5,5,gridDims), torch.linspace(-5,5,gridDims)])
    xyGrid = torch.stack([x,y],dim=2).float()
    
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,5))
    
    #Show the points which were sampled to construct the GP model
    axes[1].scatter(testPoints[:,0].detach(),testPoints[:,1].detach(),c='orange',s=8,edgecolors='black')
    axes[1].scatter(trainingPoints[:,0].detach(),trainingPoints[:,1].detach(),c='orange',s=24,edgecolors='white')