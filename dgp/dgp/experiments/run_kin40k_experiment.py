# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:57:47 2014

@author: marc
"""
#import sys

import experiments.kin40k_experiment as kin40

randomSeed = [1]
numExperts = [4]

for i in randomSeed:
  for j in numExperts:
    kin40.kin40k_experiment(i,j)
#    print str(i),str(j)
#    sys.argv = [str(i), str(j)]
#    execfile('experiments/kin40k_experiment.py')