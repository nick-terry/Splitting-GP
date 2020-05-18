# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 21:16:52 2015

@author: marc
"""


import pandas
import cPickle as pk
import numpy as np
from hgp import DataSet as DS
from os import listdir
from os.path import isfile, join
from hgp.utils.analyze import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
matplotlib.rcParams['text.usetex']=True

# set some default values for plotting
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']  =14



def getData(fn):
    numbers = []
    #f=open('data/kin-32nh/Dataset.data','r')
    f=open(fn, 'r')
    for eachLine in f:
        eachLine = eachLine.strip()
        y = [float(value) for value in eachLine.split()]
        numbers.append(y)
    f.close()
    return numbers

ytest = np.asarray(getData('./data/kin40K/kin40k_test_labels.asc')).squeeze()

mypath = 'results/kin40k/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]


rmse_bcm4 = []
rmse_gbcm4 = []
rmse_poe4 = []
rmse_gpoe4 = []
rmse_sor4 = []

nlpd_bcm4 = []
nlpd_gbcm4 = []
nlpd_poe4 = []
nlpd_gpoe4 = []
nlpd_sor4 = []

rmse_bcm16 = []
rmse_gbcm16 = []
rmse_poe16 = []
rmse_gpoe16 = []
rmse_sor16 = []

nlpd_bcm16 = []
nlpd_gbcm16 = []
nlpd_poe16 = []
nlpd_gpoe16 = []
nlpd_sor16 = []

rmse_bcm64 = []
rmse_gbcm64 = []
rmse_poe64 = []
rmse_gpoe64 = []
rmse_sor64 = []

nlpd_bcm64 = []
nlpd_gbcm64 = []
nlpd_poe64 = []
nlpd_gpoe64 = []
nlpd_sor64 = []

rmse_bcm256 = []
rmse_gbcm256= []
rmse_poe256= []
rmse_gpoe256 = []
rmse_sor256 = []

nlpd_bcm256= []
nlpd_gbcm256= []
nlpd_poe256= []
nlpd_gpoe256= []
nlpd_sor256= []

nlml_time4 = []; nlml_time16 = []; nlml_time64 = []; nlml_time256=[]

for filename in onlyfiles:
  with open(mypath+filename,'r') as f:
    results = pk.load(f)
    
    # load predictions
    ybcm = np.asarray(results['ybcm']).reshape(-1,1).squeeze()
    ypoe = np.asarray(results['ypoe']).reshape(-1,1).squeeze()
    ygpoe = np.asarray(results['ygpoe']).reshape(-1,1).squeeze()
    ygbcm = np.asarray(results['ygbcm']).reshape(-1,1).squeeze()
    ysor = np.asarray(results['ysor']).reshape(-1,1).squeeze()
    sbcm = np.asarray(results['sbcm']).reshape(-1,1).squeeze()
    spoe = np.asarray(results['spoe']).reshape(-1,1).squeeze()
    sgpoe = np.asarray(results['sgpoe']).reshape(-1,1).squeeze()
    sgbcm = np.asarray(results['sgbcm']).reshape(-1,1).squeeze()
    ssor = np.asarray(results['ssor']).reshape(-1,1).squeeze()
    
    if results['numExperts']==4:
      
      rmse_bcm4 += (rmse(ytest,ybcm),)
      rmse_poe4 += (rmse(ytest,ypoe),)
      rmse_gpoe4 += (rmse(ytest,ygpoe),)
      rmse_gbcm4 += (rmse(ytest,ygbcm),)
      rmse_sor4 += (rmse(ytest,ysor),)
      
      nlpd_bcm4 += (np.mean(nlpd(ybcm, sbcm, ytest)),)
      nlpd_poe4 += (np.mean(nlpd(ypoe, spoe, ytest)),)
      nlpd_gpoe4 += (np.mean(nlpd(ygpoe, sgpoe, ytest)),)
      nlpd_gbcm4 += (np.mean(nlpd(ygbcm, sgbcm, ytest)),)
      nlpd_sor4 += (np.mean(nlpd(ysor, ssor, ytest)),)
      
      nlml_time4 += (results['NLML_gradient-time'],)
    
    if results['numExperts']==16:
      
      rmse_bcm16 += (rmse(ytest,ybcm),)
      rmse_poe16 += (rmse(ytest,ypoe),)
      rmse_gpoe16 += (rmse(ytest,ygpoe),)
      rmse_gbcm16 += (rmse(ytest,ygbcm),)
      rmse_sor16 += (rmse(ytest,ysor),)
      
      nlpd_bcm16 += (np.mean(nlpd(ybcm, sbcm, ytest)),)
      nlpd_poe16 += (np.mean(nlpd(ypoe, spoe, ytest)),)
      nlpd_gpoe16 += (np.mean(nlpd(ygpoe, sgpoe, ytest)),)
      nlpd_gbcm16 += (np.mean(nlpd(ygbcm, sgbcm, ytest)),)
      nlpd_sor16 += (np.mean(nlpd(ysor, ssor, ytest)),)
      
      nlml_time16 += (results['NLML_gradient-time'],)
      
    if results['numExperts']==64:
      
      rmse_bcm64 += (rmse(ytest,ybcm),)
      rmse_poe64 += (rmse(ytest,ypoe),)
      rmse_gpoe64 += (rmse(ytest,ygpoe),)
      rmse_gbcm64 += (rmse(ytest,ygbcm),)
      rmse_sor64 += (rmse(ytest,ysor),)
      
      nlpd_bcm64 += (np.mean(nlpd(ybcm, sbcm, ytest)),)
      nlpd_poe64 += (np.mean(nlpd(ypoe, spoe, ytest)),)
      nlpd_gpoe64 += (np.mean(nlpd(ygpoe, sgpoe, ytest)),)
      nlpd_gbcm64 += (np.mean(nlpd(ygbcm, sgbcm, ytest)),)
      nlpd_sor64 += (np.mean(nlpd(ysor, ssor, ytest)),)
      
      nlml_time64 += (results['NLML_gradient-time'],)
      
    if results['numExperts']==256:
      
      rmse_bcm256 += (rmse(ytest,ybcm),)
      rmse_poe256 += (rmse(ytest,ypoe),)
      rmse_gpoe256 += (rmse(ytest,ygpoe),)
      rmse_gbcm256 += (rmse(ytest,ygbcm),)
      rmse_sor256 += (rmse(ytest,ysor),)
      
      nlpd_bcm256 += (np.mean(nlpd(ybcm, sbcm, ytest)),)
      nlpd_poe256 += (np.mean(nlpd(ypoe, spoe, ytest)),)
      nlpd_gpoe256 += (np.mean(nlpd(ygpoe, sgpoe, ytest)),)
      nlpd_gbcm256 += (np.mean(nlpd(ygbcm, sgbcm, ytest)),)
      nlpd_sor256 += (np.mean(nlpd(ysor, ssor, ytest)),)
      
      nlml_time256 += (results['NLML_gradient-time'],)
 
with open('real_timings.pk','r') as f:
    real_timings = pk.load(f)   
    
nlml_time4 = real_timings['nlml_time4']
nlml_time16 = real_timings['nlml_time16']
nlml_time64 = real_timings['nlml_time64']
nlml_time256 = real_timings['nlml_time256']
   
# print some results
print '4 Experts:'

print 'Training time:', np.mean(nlml_time4)#, '+/-', np.std(nlml_time4)/np.sqrt(len(nlml_time4))

print '-----------'
print 'RMSE'
print 'gBCM:', np.mean(rmse_gbcm4), '+/-', np.std(rmse_gbcm4)/np.sqrt(len(rmse_gbcm4))
print 'BCM :', np.mean(rmse_bcm4), '+/-', np.std(rmse_bcm4)/np.sqrt(len(rmse_bcm4))
print 'gPoE:', np.mean(rmse_gpoe4), '+/-', np.std(rmse_gpoe4)/np.sqrt(len(rmse_gpoe4))
print 'PoE :', np.mean(rmse_poe4), '+/-', np.std(rmse_poe4)/np.sqrt(len(rmse_poe4))
print 'SOR :', np.mean(rmse_sor4), '+/-', np.std(rmse_sor4)/np.sqrt(len(rmse_sor4))

print '-----------'
print 'NLPD'
print 'gBCM:', np.mean(nlpd_gbcm4), '+/-', np.std(nlpd_gbcm4)/np.sqrt(len(nlpd_gbcm4))
print 'BCM :', np.mean(nlpd_bcm4), '+/-', np.std(nlpd_bcm4)/np.sqrt(len(nlpd_bcm4))
print 'gPoE:', np.mean(nlpd_gpoe4), '+/-', np.std(nlpd_gpoe4)/np.sqrt(len(nlpd_gpoe4))
print 'PoE :', np.mean(nlpd_poe4), '+/-', np.std(nlpd_poe4)/np.sqrt(len(nlpd_poe4))
print 'SOR :', np.mean(nlpd_sor4), '+/-', np.std(nlpd_sor4)/np.sqrt(len(nlpd_sor4))

print '----------------------'
print '16 Experts:'

print 'Training time:', np.mean(nlml_time16)#, '+/-', np.std(nlml_time16)/np.sqrt(len(nlml_time16))

print '-----------'
print 'RMSE'
print 'gBCM:', np.mean(rmse_gbcm16), '+/-', np.std(rmse_gbcm16)/np.sqrt(len(rmse_gbcm16))
print 'BCM :', np.mean(rmse_bcm16), '+/-', np.std(rmse_bcm16)/np.sqrt(len(rmse_bcm16))
print 'gPoE:', np.mean(rmse_gpoe16), '+/-', np.std(rmse_gpoe16)/np.sqrt(len(rmse_gpoe16))
print 'PoE :', np.mean(rmse_poe16), '+/-', np.std(rmse_poe16)/np.sqrt(len(rmse_poe16))
print 'SOR :', np.mean(rmse_sor16), '+/-', np.std(rmse_sor16)/np.sqrt(len(rmse_sor16))

print '-----------'
print 'NLPD'
print 'gBCM:', np.mean(nlpd_gbcm16), '+/-', np.std(nlpd_gbcm16)/np.sqrt(len(nlpd_gbcm16))
print 'BCM :', np.mean(nlpd_bcm16), '+/-', np.std(nlpd_bcm16)/np.sqrt(len(nlpd_bcm16))
print 'gPoE:', np.mean(nlpd_gpoe16), '+/-', np.std(nlpd_gpoe16)/np.sqrt(len(nlpd_gpoe16))
print 'PoE :', np.mean(nlpd_poe16), '+/-', np.std(nlpd_poe16)/np.sqrt(len(nlpd_poe16))
print 'SOR :', np.mean(nlpd_sor16), '+/-', np.std(nlpd_sor16)/np.sqrt(len(nlpd_sor16))

print '----------------------'

# print some results
print '64 Experts:'
print 'Training time:',  np.mean(nlml_time64)#, '+/-', np.std(nlml_time64)/np.sqrt(len(nlml_time64))

print '-----------'
print 'RMSE'
print 'gBCM:', np.mean(rmse_gbcm64), '+/-', np.std(rmse_gbcm64)/np.sqrt(len(rmse_gbcm64))
print 'BCM :', np.mean(rmse_bcm64), '+/-', np.std(rmse_bcm64)/np.sqrt(len(rmse_bcm64))
print 'gPoE:', np.mean(rmse_gpoe64), '+/-', np.std(rmse_gpoe64)/np.sqrt(len(rmse_gpoe64))
print 'PoE :', np.mean(rmse_poe64), '+/-', np.std(rmse_poe64)/np.sqrt(len(rmse_poe64))
print 'SOR :', np.mean(rmse_sor64), '+/-', np.std(rmse_sor64)/np.sqrt(len(rmse_sor64))

print '-----------'
print 'NLPD'
print 'gBCM:', np.mean(nlpd_gbcm64), '+/-', np.std(nlpd_gbcm64)/np.sqrt(len(nlpd_gbcm64))
print 'BCM :', np.mean(nlpd_bcm64), '+/-', np.std(nlpd_bcm64)/np.sqrt(len(nlpd_bcm64))
print 'gPoE:', np.mean(nlpd_gpoe64), '+/-', np.std(nlpd_gpoe64)/np.sqrt(len(nlpd_gpoe64))
print 'PoE :', np.mean(nlpd_poe64), '+/-', np.std(nlpd_poe64)/np.sqrt(len(nlpd_poe64))
print 'SOR :', np.mean(nlpd_sor64), '+/-', np.std(nlpd_sor64)/np.sqrt(len(nlpd_sor64))

print '----------------------'

# print some results
print '256 Experts:'
print 'Training time:', np.mean(nlml_time256)#, '+/-', np.std(nlml_time256)/np.sqrt(len(nlml_time256))

print '-----------'
print 'RMSE'
print 'gBCM:', np.mean(rmse_gbcm256), '+/-', np.std(rmse_gbcm256)/np.sqrt(len(rmse_gbcm256))
print 'BCM :', np.mean(rmse_bcm256), '+/-', np.std(rmse_bcm256)/np.sqrt(len(rmse_bcm256))
print 'gPoE:', np.mean(rmse_gpoe256), '+/-', np.std(rmse_gpoe256)/np.sqrt(len(rmse_gpoe256))
print 'PoE :', np.mean(rmse_poe256), '+/-', np.std(rmse_poe256)/np.sqrt(len(rmse_poe256))
print 'SOR :', np.mean(rmse_sor256), '+/-', np.std(rmse_sor256)/np.sqrt(len(rmse_sor256))

print '-----------'
print 'NLPD'
print 'gBCM:', np.mean(nlpd_gbcm256), '+/-', np.std(nlpd_gbcm256)/np.sqrt(len(nlpd_gbcm256))
print 'BCM :', np.mean(nlpd_bcm256), '+/-', np.std(nlpd_bcm256)/np.sqrt(len(nlpd_bcm256))
print 'gPoE:', np.mean(nlpd_gpoe256), '+/-', np.std(nlpd_gpoe256)/np.sqrt(len(nlpd_gpoe256))
print 'PoE :', np.mean(nlpd_poe256), '+/-', np.std(nlpd_poe256)/np.sqrt(len(nlpd_poe256))
print 'SOR :', np.mean(nlpd_sor256), '+/-', np.std(nlpd_sor256)/np.sqrt(len(nlpd_sor256))

rmse_gp = 0.10784458037066989
nlpd_gp = -3.3357456748175092
time_gp= 1161

#real_timings={}
#real_timings['nlml_time4'] = np.mean(nlml_time4)
#real_timings['nlml_time64'] = np.mean(nlml_time64)
#real_timings['nlml_time16'] = np.mean(nlml_time16)
#real_timings['nlml_time256'] = np.mean(nlml_time256)
#real_timings['time_gp'] = time_gp


rmse_gbcm = [rmse_gp,np.mean(rmse_gbcm4), np.mean(rmse_gbcm16), np.mean(rmse_gbcm64),  np.mean(rmse_gbcm256)]
rmse_bcm = [rmse_gp, np.mean(rmse_bcm4),np.mean(rmse_bcm16), np.mean(rmse_bcm64),  np.mean(rmse_bcm256)]
rmse_gpoe = [rmse_gp, np.mean(rmse_gpoe4),np.mean(rmse_gpoe16), np.mean(rmse_gpoe64),  np.mean(rmse_gpoe256)]
rmse_poe = [rmse_gp, np.mean(rmse_poe4),np.mean(rmse_poe16), np.mean(rmse_poe64),  np.mean(rmse_poe256)]
rmse_sor = [rmse_gp, np.mean(rmse_sor4), np.mean(rmse_sor16), np.mean(rmse_sor64),  np.mean(rmse_sor256)]

nlpd_gbcm = [nlpd_gp,np.mean(nlpd_gbcm4),  np.mean(nlpd_gbcm16), np.mean(nlpd_gbcm64),  np.mean(nlpd_gbcm256)]
nlpd_bcm = [nlpd_gp, np.mean(nlpd_bcm4), np.mean(nlpd_bcm16), np.mean(nlpd_bcm64),  np.mean(nlpd_bcm256)]
nlpd_gpoe = [nlpd_gp,np.mean(nlpd_gpoe4),np.mean(nlpd_gpoe16), np.mean(nlpd_gpoe64),  np.mean(nlpd_gpoe256)]
nlpd_poe = [nlpd_gp, np.mean(nlpd_poe4), np.mean(nlpd_poe16), np.mean(nlpd_poe64),  np.mean(nlpd_poe256)]
nlpd_sor = [nlpd_gp,  np.mean(nlpd_sor4),np.mean(nlpd_sor16), np.mean(nlpd_sor64),  np.mean(nlpd_sor256)]

nlpd_sor = [nlpd_gp, 0.36, 0.83, 0.95, 1.05]
### plotting
xticks = [1,4,16,64,256]
xticks2 = [10000, 2500, 625, 156, 39] # data points per expert
xtickstimeBCM = [np.mean(nlml_time256), np.mean(nlml_time64), np.mean(nlml_time16), np.mean(nlml_time4), time_gp]
xtickstimeSOR = [0.24, 0.50, 1.83, 27.20, time_gp]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticks(xticks)
ax.set_ylabel('RMSE')
ax.set_xlabel('#Experts')
plt.semilogx(xticks, rmse_gbcm, 'ob-', label="gBCM")
plt.semilogx(xticks, rmse_bcm, 'sr-', label="BCM")
plt.semilogx(xticks, rmse_gpoe, '.m-', label="gPoE")
plt.semilogx(xticks, rmse_poe, '*c--', label="PoE")
plt.semilogx(xticks, rmse_sor, 'dg-', label="SOR")
plt.semilogx(xticks, np.ones([len(xticks),1])*rmse_gp, 'k--', label="GP")
handles, labels = ax.get_legend_handles_labels()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
           
nlpd_gbcm.reverse()  
nlpd_bcm.reverse()  
nlpd_gpoe.reverse()  
nlpd_poe.reverse()  
nlpd_sor.reverse()  

rmse_gbcm.reverse()  
rmse_bcm.reverse()  
rmse_gpoe.reverse()  
rmse_poe.reverse()  
rmse_sor.reverse()  
         
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticks(xticks2)
ax.set_ylabel('NLPD')
ax.set_xlabel('#Data points per expert')
plt.semilogx(xticks2, nlpd_gbcm , 'ob-', label="gBCM")
plt.semilogx(xticks2, nlpd_bcm, 'sr-', label="BCM")
plt.semilogx(xticks2, nlpd_gpoe, '.m-', label="gPoE")
plt.semilogx(xticks2, nlpd_poe, '*c--', label="PoE")
plt.plot(xticks2, nlpd_sor, 'dg-', label="SOD")
plt.plot(xticks2, np.ones([len(xticks2),1])*nlpd_gp, 'k--', label="GP")
handles, labels = ax.get_legend_handles_labels()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylim([-4.5, 3])
plt.xlim([39,10000])


numPoints = [39,156,625,2500,10000]
## plot vs training time
pp = PdfPages('time_vs_nlpd.pdf')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticks(xtickstimeBCM)
ax.set_ylabel('NLPD', fontsize=18)
ax.set_xlabel('#Gradient time in sec', fontsize=18)
ax.semilogx(xtickstimeBCM, nlpd_gbcm , 'ob-', label="gBCM",markersize=10)
ax.semilogx(xtickstimeBCM, nlpd_bcm, 'sr-', label="BCM",markersize=10)
ax.semilogx(xtickstimeBCM, nlpd_gpoe, '.m-', label="gPoE",markersize=10)
ax.semilogx(xtickstimeBCM, nlpd_poe, '*c--', label="PoE",markersize=10)
ax.plot(xtickstimeSOR, nlpd_sor, 'dg-', label="SOD",markersize=10)
ax.plot(xtickstimeBCM, np.ones([len(xticks2),1])*nlpd_gp, 'k--', label="GP")
handles, labels = ax.get_legend_handles_labels()
plt.legend(loc='best', borderaxespad=0, frameon=False)
plt.ylim([-3.5,2])
ax.set_xlim(0,1170)
bx = ax.twiny()
bx.set_xscale('log')  
bx.set_xlim(39,1170)
bx.set_xticks(xtickstimeBCM)
bx.set_xticklabels(numPoints)
bx.set_xlabel('#Points/Expert',fontsize=18)
#bx.set_xticklabels(tick_function(x)) # Convert the Data-x-coordinates of the first x-axes to the Desired x', with the tick_function(X)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(pp, format='pdf')
pp.close()


pp = PdfPages('time_vs_rmse.pdf')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel('RMSE', fontsize=18)
ax.set_xlabel('#Gradient time in sec', fontsize=18)
ax.semilogx(xtickstimeBCM, rmse_gbcm , 'ob-', label="gBCM",markersize=10)
ax.semilogx(xtickstimeBCM, rmse_bcm, 'sr-', label="BCM",markersize=10)
ax.semilogx(xtickstimeBCM, rmse_gpoe, '.m-', label="gPoE",markersize=10)
ax.semilogx(xtickstimeBCM, rmse_poe, '*c--', label="PoE",markersize=10)
ax.semilogx(xtickstimeSOR, rmse_sor, 'dg-', label="SOD",markersize=10)
ax.semilogx(xtickstimeBCM, np.ones([len(xticks2),1])*rmse_gp, 'k--', label="GP")
#ax.set_xticks(xtickstimeBCM)
#ax.set_xticklabels(xtickstimeBCM)
handles, labels = ax.get_legend_handles_labels()
plt.legend(loc='best', borderaxespad=0, frameon=False)
plt.ylim([0,0.8])
ax.set_xlim(0,1170)
bx = ax.twiny()
bx.set_xscale('log')  
bx.set_xlim(39,1170)
bx.set_xticks(xtickstimeBCM)
bx.set_xticklabels(numPoints)
#bx.set_xlim(0, xtickstimeBCM[-1])

bx.set_xlabel('#Points/Expert',fontsize=18)
#bx.set_xticklabels(tick_function(x)) # Convert the Data-x-coordinates of the first x-axes to the Desired x', with the tick_function(X)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(pp, format='pdf')
pp.close()