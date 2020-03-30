# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:54:30 2020

@author: pnter
"""

import pandas as pd
import os
from os.path import isfile,join
import matplotlib.pyplot as plt

folderPath = 'C:/Users/pnter/Documents/GitHub/GP-Regression-Research/ExperimentResults'

def loadFiles():
    #Get only files with .csv extension
    files = [path for path in os.listdir(folderPath) if isfile(join(folderPath,path)) 
             and path[-4:]=='.csv']
    dataframes = []
    
    for file in files:
        dataframe = pd.read_csv(file)
        metadata = file.split('-')
        dataframe['response_function'] = metadata[-2]
        dataframe['noise'] = 1 if metadata[-1]=='noise.csv' else 0
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
models = ['splitting','local','exact']
metrics = ['avg_mse','avg_memory_usage','avg_training_time']
for metric in metrics:
    series[metric] = []
    
for model in models:
    toyFnNoiseData = df.loc[(df['model']==model)&(df['response_function']=='bimodal')&(df['noise']==1)]
    stats = getSummaryStats(toyFnNoiseData)
    for metric in metrics:
        series[metric].append(stats[0][metric])

observations_vals = df['observations'].unique()

for metric in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(observations_vals,series[metric][0],'-o',
            observations_vals,series[metric][1],'-o',
            observations_vals,series[metric][2],'-o')
    ax.legend(models)
    ax.set_title(metric)
    ax.axvline(1200)
    
for metric in ['avg_mse']:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(observations_vals[:-1],series[metric][0][:-1],
            observations_vals[:-1],series[metric][1][:-1],
            observations_vals[:-1],series[metric][2][:-1])
    ax.legend(models)
    ax.set_title(metric)