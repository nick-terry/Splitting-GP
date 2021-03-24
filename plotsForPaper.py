# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:13:19 2020

@author: pnter
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Rescale to values between 0 and 1 
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)


#kin40_plot()
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
ax.set_ylabel('RMSE',fontsize='x-large')
ax.set_xlabel('Observations per expert/local model',fontsize='x-large')
ax2.set_xlabel('w_gen',fontsize='x-large')

plt.savefig('plot_kin40_comparison_nonan.pdf',dpi=300)

#powergen plot
powergen_rbcm = pd.read_csv('C:/Users/pnter/Documents/GitHub/GP-Regression-Research/rBCM_powergen_results_ppe_newformat.csv')
powergen_splitting = pd.read_csv('C:/Users/pnter/Documents/GitHub/GP-Regression-Research/powergen_results_splitting_10reps.csv')

#log transform
'''
powergen_rbcm['time'] = np.log(powergen_rbcm['time']) 
powergen_splitting = np.log(powergen_splitting['time']) 
'''

paramsStrings = powergen_splitting['params']
splittingLimits = []
for string in paramsStrings:
        splittingLimit = string.split(',')[1][-4:]
        if splittingLimit[0]==':':
            splittingLimit = splittingLimit[1:]
        splittingLimit = splittingLimit.strip()
        splittingLimits.append(int(splittingLimit))

powergen_splitting['params'] = splittingLimits

splittingMeans = powergen_splitting.groupby('params').mean().reset_index()
splittingStd = powergen_splitting.groupby('params').std().reset_index()

splittingMeans = splittingMeans.sort_values('params',ascending=False)
splittingStd = splittingStd.sort_values('params',ascending=False)
splittingCIHW = .96/np.sqrt(10)*splittingStd


rbcmMeans = powergen_rbcm.groupby('numChildren').mean().reset_index()
rbcmStd = powergen_rbcm.groupby('numChildren').std().reset_index()
rbcmCIHW = .96/np.sqrt(10)*rbcmStd

'''
powerGenRMSE_CIHW = .96*np.sqrt(10)*powergen_rbcm['std_rmse']
powerGenTime_CIHW = .96*np.sqrt(10)*powergen_rbcm['std_time']
'''

powerFig = plt.figure()
powerAxes = powerFig.subplots(2)

#number of observations used for training
numObs= 7622

powerAxes[0].plot(splittingMeans['params'],splittingMeans['rmse'],'-o',color=tableau20[0])
powerAxes[0].fill_between(splittingMeans['params'],
                          splittingMeans['rmse']-splittingCIHW['rmse'],
                          splittingMeans['rmse']+splittingCIHW['rmse'],
                          color=tableau20[0],alpha=.7)
powerAxes[0].set_xlim(powerAxes[0].get_xlim()[::-1])

powerAxes[0].plot(numObs/rbcmMeans['numChildren'],rbcmMeans['rmse'],'-o',color=tableau20[2])
powerAxes[0].fill_between(numObs/rbcmMeans['numChildren'],
                          rbcmMeans['rmse']-rbcmCIHW['rmse'],
                          rbcmMeans['rmse']+rbcmCIHW['rmse'],
                          color=tableau20[2],alpha=.7)

powerAxes[0].set_ylabel('RMSE',fontsize='x-large')
powerAxes[0].set_xlabel('Observations per expert/local model',fontsize='x-large')
powerAxes[0].set_yticks(list(range(4,10,1)))

powerAxes[0].legend(['splitting','rbcm'])

powerAxes[1].plot(splittingMeans['params'],splittingMeans['time'],'-o',color=tableau20[0])
powerAxes[1].fill_between(splittingMeans['params'],
                          splittingMeans['time']+splittingCIHW['time'],
                          splittingMeans['time']-splittingCIHW['time'],
                          color=tableau20[0],alpha=.7)
powerAxes[1].set_xlim(powerAxes[1].get_xlim()[::-1])

powerAxes[1].plot(numObs/rbcmMeans['numChildren'],rbcmMeans['time'],'-o',color=tableau20[2])
powerAxes[1].fill_between(numObs/rbcmMeans['numChildren'],
                          rbcmMeans['time']-rbcmCIHW['time'],
                          rbcmMeans['time']+rbcmCIHW['time'],
                          color=tableau20[2],alpha=.7)

powerAxes[1].set_ylabel('Runtime (Sec)',fontsize='x-large')
powerAxes[1].set_xlabel('Observations per expert/local model',fontsize='x-large')
powerAxes[1].set_yticks(list(range(0,150,30)))

powerFig.savefig('powergen.pdf',dpi=300)