# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:54:30 2020

@author: pnter
"""

import pandas as pd
import os
from os.path import isfile,join
import matplotlib.pyplot as plt
import numpy as np

folderPath = 'C:/Users/pnter/Documents/GitHub/GP-Regression-Research/ExperimentResults'

def loadFiles():
    #Get only files with .csv extension
    files = [path for path in os.listdir(folderPath) if isfile(join(folderPath,path)) 
             and path[-4:]=='.csv' and not 'BACKUP' in path]
    dataframes = []
    
    for file in files:
        dataframe = pd.read_csv(file)
        metadata = file.split('-')
        dataframe['response_function'] = metadata[-2]
        dataframe['noise'] = 1 if metadata[-1]=='noise.csv' else 0
        dataframe['mean'] = 'zero' if 'zeromean' in metadata else 'constant'
        dataframes.append(dataframe)
    
    dataframe = pd.concat(dataframes)
    dataframe.rename(columns={'Unnamed: 0':'observations'},inplace=True)
    return dataframe

#Compute summary statistics for the runs i.e. Mean, Std, Confidence intervals
def getSummaryStats(df):
    df = df.groupby(['model','params','response_function','noise','observations'])
    return df.mean(),df.std()

df = loadFiles()

series = {}
#models = ['splitting','local','exact']
models = ['splitting']
metrics = ['avg_mse','avg_memory_usage','avg_training_time']
ylabels = ['MSE','Memory Usage (Kb)','Training Time (sec)']
for metric in metrics:
    series[metric] = {}
    
    for stat in ['mean','std','cihw']:
        series[metric][stat] = []
        
for model in models:
    #toyFnNoiseData = df.loc[(df['model']==model)&(df['response_function']=='bimodal')&(df['noise']==1)&(df['params']=='fantasyUpdate=False')]
    toyFnNoiseData = df.loc[(df['model']==model)&(df['response_function']=='toyfn')&(df['noise']==1)&(df['mean']=='constant')]
    stats = getSummaryStats(toyFnNoiseData)
    for metric in metrics:
        #series[metric].append(stats[0][metric][:25])
        series[metric]['mean'].append(stats[0][metric])
        series[metric]['std'].append(stats[1][metric])
        #Compute half-width of the 95% CI
        cihw = .96 * stats[1][metric] / np.sqrt(10)
        series[metric]['cihw'].append(cihw)
        
observations_vals = df['observations'].unique()
#observations_vals = df['observations'].unique()

for metric,ylabel in zip(metrics,ylabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for index in range(len(series[metric]['mean'])):
        
        mean = series[metric]['mean'][index]
        cihw = series[metric]['cihw'][index]
        
        ax.plot(observations_vals,mean,'-o')
        ax.fill_between(observations_vals,mean-cihw,mean+cihw,alpha=.7)
    '''
    mean = series[metric]['mean'][0]
    cihw = series[metric]['cihw'][0]
    ax.plot(observations_vals,mean,'-o')
    ax.fill_between(observations_vals,mean-cihw,mean+cihw,alpha=.5)
    '''
    
    ax.legend(models)
    ax.set_title(metric)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('# Observations')
    #ax.axvline(1200,color='orange')
'''
for metric in ['avg_mse']:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(observations_vals[:-4],series[metric][0][:-4],'-o',
            observations_vals[:-4],series[metric][1][:-4],'-o',
            observations_vals[:-4],series[metric][2][:-4],'-o')
    ax.legend(models)
    ax.set_title(metric)
    ax.set_ylabel('MSE')
    ax.set_xlabel('# Observations')
'''