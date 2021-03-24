# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:41:59 2020

@author: pnter
"""


import sklearn as skl
import pytorch
import numpy as np

'''
Implement a SplittingLocalGP using sklearn instead of gpytorch
'''
class SplittingLocalGP:
    
    def __init__(self):
        
        self.splittingLimit = splittingLimit