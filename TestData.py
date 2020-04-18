# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:32:42 2020

@author: pnter
"""

import torch
import pandas as pd

def icethick(full=False):
    df = pd.read_csv('ICETHK.csv')
    if full:
        return df
    
    else:
        df = df[['Longitude W','Latitude S','Thickness (m)']]
        
        #Transform into PyTorch tensors of predictors and responses
        x = torch.tensor(df[['Longitude W','Latitude S']].to_numpy())
        y = torch.tensor(df['Thickness (m)'].to_numpy())
        
        #Get every 3rd element for now to test on smaller scale
        x = x[::3,:].float()
        y = y[::3].float()
        y = y/torch.max(y) #Scale down for regression
        
        return x,y
