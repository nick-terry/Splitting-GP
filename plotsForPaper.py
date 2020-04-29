# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:13:19 2020

@author: pnter
"""


import matplotlib.pyplot as plt
import pandas as pd

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Rescale to values between 0 and 1 
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)

kin40_splitting = pd.read_csv('kin40_results_splitting_compare_to_rbcm_m1.csv')
kin40_local = pd.read_csv('kin40_results_local_compare.csv')
kin40_rbcm = pd.read_csv('RBCM.csv')

fig = plt.figure()
ax = fig.add_subplot(111)

splittingParams = kin40_splitting['params']
splittingPPE = [int(splittingParam.split('\'')[4][2:][:-2]) for splittingParam in splittingParams]
splittingRMSE = kin40_splitting['rmse']

ax.plot(splittingPPE,splittingRMSE,'-o',color=tableau20[0])

rbcmPPE = kin40_rbcm['npoints per expert']
rbcmRMSE = kin40_rbcm['rmse']

ax.plot(rbcmPPE,rbcmRMSE,'-o',color=tableau20[2])

ax.axhline(rbcmRMSE[4],ls='--',color=tableau20[8])

localWGEN = kin40_local['params']
localWGEN = [float(params.split(',')[1][-6:].strip()) for params in localWGEN]
localRMSE = kin40_local['rmse']

ax2 = ax.twiny()
ax2.set_xscale('log')
lins2 = ax2.plot(localWGEN[:-1],localRMSE[:-1],'-o',color=tableau20[4])
ax2.set_xlim(ax2.set_xlim()[::-1])

# add lines to make legend
lns = ax.lines[:-1]+lins2+[ax.lines[-1]]
labs = [l.get_label() for l in lns]
ax2.legend(lns, ['splitting','rbcm','local','full GP'], loc=0)

#label axes
ax.set_ylabel('RMSE')
ax.set_xlabel('Observations per expert/local model')
ax2.set_xlabel('w_gen')

plt.savefig('plot_kin40_comparison_nonan.png',dpi=300)

#kin40_plot()