# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:40:51 2020

@author: pnter
"""
import warnings
from sys import platform

if platform == 'linux':
    
    #This module does not exist on Windows installations. Should work on Hyak
    import resource
    
    def getMemoryUsage():
        #Return the memory usage of this process and all child processes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    
else:
    print('Memory usage cannot be logged on a {0} machine.'.format(platform))
    
    def getMemoryUsage():
        print('Memory usage cannot be logged on a {0} machine.'.format(platform))
        return 0