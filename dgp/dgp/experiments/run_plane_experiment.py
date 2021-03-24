# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:57:47 2014

@author: marc
"""
#import sys
import experiments.plane_experiment as plane

randomSeed = 100
train_N = 700000

plane.plane_experiment(randomSeed, train_N, 100000)
