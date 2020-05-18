# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:45:23 2015

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

with open('data/plane/filtered_data.pickle','r') as f:
    data = pk.load(f)

y = data['ArrDelay']
X = data[['Month','DayofMonth','DayOfWeek','DepTime','ArrTime','AirTime','Distance','plane_age']]

# hgp0 = hgp.HGP(X,y,pool='default',profile=[(100,'random','rep2')]*3) # create GP with default cov

data = DS(X,y)


mypath = 'results/plane/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

#filename = mypath+'2015-01-29-02-59-38_results_plane_700000_16_3_31.pk'
#
#with open(filename,'r') as f:
#    results = pk.load(f)


##################
#    i_s = results['i_s']
#    ytest = np.asarray(data.y)[i_s]
#    Xtest = np.asarray(data.X)[i_s,:]
#    loghypers = results['gbcm_params']
#    keyCovariate = np.argmin(loghypers[range(8)])
#    labels= ['Month','Day of Month','Day of Week','DepTime','ArrTime','AirTime','Distance','PlaneAge']
#    length_scales = np.exp(loghypers[range(8)]) 
#    
#    
#    pp = PdfPages('multipage.pdf')
#    fig = plt.figure()
#    fig.subplots_adjust(bottom=0.2)
#    
#    ax = fig.add_subplot(111)
#    rects1 = ax.bar(np.arange(8), 1./length_scales, 0.7, color='red')
#    ax.set_ylabel('Importance')
#    ax.set_xticklabels(labels)
#    xTickMarks = labels
#    xtickNames = ax.set_xticklabels(xTickMarks)
#    plt.setp(xtickNames, rotation=45, fontsize=10)
#    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    #plt.autoscale(enable=True, tight=True)
#    #plt.show()
#    plt.savefig(pp, format='pdf')
#    pp.close()
#    
#    
#    xtest = np.zeros(shape=(len(Xtest),2))
#    xtest[:,0] = Xtest[:,keyCovariate]    
#    for hour in range(24):
#      for n in range(len(xtest)):
#        if (xtest[n,0] > hour*100.0) and (xtest[n,0] <= (hour+1)*100.0):
#          xtest[n,1] = hour
#     
#    ygbcm = np.asarray(results['ygbcm']).reshape(-1,1)
#    sgbcm = np.asarray(results['sgbcm']).reshape(-1,1)
#    # ground truth heuristic
#    ground_truth_mean = np.zeros([24,1])
#    ground_truth_std = np.zeros([24,1])
#    gbcm_mean = np.zeros([24,1])
#    gbcm_std = np.zeros([24,1])
#    for hour in range(24):
#      ground_truth_mean[hour] = np.mean(ytest[xtest[:,1]==hour])
#      ground_truth_std[hour] = np.std(ytest[xtest[:,1]==hour])
#      gbcm_mean[hour] = np.mean(ygbcm[xtest[:,1]==hour])
#      gbcm_std[hour] = np.sqrt(np.mean(sgbcm[xtest[:,1]==hour])+np.var(ygbcm[xtest[:,1]==hour]))
#      
#      
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    width = 0.35
#    rects1 = ax.bar(np.arange(24), ground_truth_mean.squeeze(), width, color='black', yerr=ground_truth_std.squeeze(), 
#                    error_kw=dict(elinewidth=2,ecolor='red'))
#
#    rects2 = ax.bar(np.arange(24)+width, gbcm_mean.squeeze(), width, color='red', yerr=gbcm_std.squeeze(),
#                    error_kw=dict(elinewidth=2,ecolor='black'))   
#        # axes and labels
#    ax.set_xlim(-width,24+width)
#    ax.set_ylim(0,45)
#    ax.set_ylabel('Delay')
#    #ax.set_title('Scores by group and gender')
#    #xTickMarks = ['Group'+str(i) for i in range(1,6)]
#    ax.set_xticks(np.arange(24)+width)
#    #xtickNames = ax.set_xticklabels(xTickMarks)
#    #plt.setp(xtickNames, rotation=45, fontsize=10)
#    
#    ## add a legend
#    ax.legend( (rects1[0], rects2[0]), ('Ground Truth', 'gBCM') )
#    
#    plt.show()
#                
#    ygbcm = np.asarray(results['ygbcm']).reshape(-1,1)
#    sgbcm = np.asarray(results['sgbcm']).reshape(-1,1)
#    plt.plot(xtest[:,1], ytest, 'ro') 
#    plt.plot(xtest[:,1], ygbcm + 2*np.sqrt(sgbcm), 'b+')
#    plt.plot(xtest[:,1], ygbcm - 2*np.sqrt(sgbcm), 'b+')
#
#
#
#    #sort 
#    sort_index = np.argsort(xtest)
#    xtest = xtest[sort_index]
#    ytest = ytest[sort_index]
#    ygbcm = ygbcm[sort_index]
#    sgbcm = sgbcm[sort_index]
#    
#    
#    plt.plot(xtest, ygbcm, 'ro')
#    plt.plot(xtest,ytest,'bo')
#    plt

################

batch_size = 1000 

rmse_gbcm700K = []; nlpd_gbcm700K = []; se_gbcm700K = []; nll_gbcm700K = []
rmse_bcm700K = []; nlpd_bcm700K = [];se_bcm700K = []; nll_bcm700K = []
rmse_gpoe700K = []; nlpd_gpoe700K = []; se_gpoe700K = []; nll_gpoe700K = []
rmse_poe700K = []; nlpd_poe700K = [];se_poe700K = []; nll_poe700K = []
rmse_sor700K = []; nlpd_sor700K = [];se_sor700K = []; nll_sor700K = []

rmse_gbcm2M = []; nlpd_gbcm2M = []; se_gbcm2M = []; nll_gbcm2M = []
rmse_bcm2M = []; nlpd_bcm2M = [];se_bcm2M = []; nll_bcm2M = []
rmse_gpoe2M = []; nlpd_gpoe2M = []; se_gpoe2M = []; nll_gpoe2M = []
rmse_poe2M = []; nlpd_poe2M = [];se_poe2M = []; nll_poe2M = []
rmse_sor2M = []; nlpd_sor2M = [];se_sor2M = []; nll_sor2M = []

rmse_gbcm5M = []; nlpd_gbcm5M = []; se_gbcm5M = []; nll_gbcm5M = []
rmse_bcm5M = []; nlpd_bcm5M = [];se_bcm5M = []; nll_bcm5M = []
rmse_gpoe5M = []; nlpd_gpoe5M = []; se_gpoe5M = []; nll_gpoe5M = []
rmse_poe5M = []; nlpd_poe5M = [];se_poe5M = []; nll_poe5M = []
rmse_sor5M = []; nlpd_sor5M = [];se_sor5M = []; nll_sor5M = []

# 700K prelim. results
add_rmse_gbcm700K = []; 
add_rmse_bcm700K = []; 
add_rmse_gpoe700K = []; 
add_rmse_poe700K = []; 
add_rmse_sor700K = []; 

add_nlpd_gbcm700K = []; 
add_nlpd_bcm700K = [];
add_nlpd_gpoe700K = []; 
add_nlpd_poe700K = [];
add_nlpd_sor700K = [];


# 2M prelim. results
add_rmse_gbcm2M = [];
add_rmse_bcm2M = []; 
add_rmse_gpoe2M = [];
add_rmse_poe2M = [];
add_rmse_sor2M = []; 

add_nlpd_gbcm2M = [];
add_nlpd_bcm2M = [];
add_nlpd_gpoe2M = []; 
add_nlpd_poe2M = []; 
add_nlpd_sor2M = []; 


# 5M prelim. results 
add_rmse_gbcm5M = []; 
add_rmse_bcm5M = []; 
add_rmse_gpoe5M = []; 
add_rmse_poe5M = [];
add_rmse_sor5M = []; 

add_nlpd_gbcm5M = [];
add_nlpd_bcm5M = [];
add_nlpd_gpoe5M = [];
add_nlpd_poe5M = [];
add_nlpd_sor5M = []; 


## 700K prelim. results
#add_rmse_gbcm700K = []; 
#add_rmse_bcm700K = []; 
#add_rmse_gpoe700K = []; 
#add_rmse_poe700K = []; 
#add_rmse_sor700K = []; 
#
#add_nlpd_gbcm700K = []; 
#add_nlpd_bcm700K = [];
#add_nlpd_gpoe700K = []; 
#add_nlpd_poe700K = [];
#add_nlpd_sor700K = [];
#
#
## 2M prelim. results
#rmse_gbcm2M = [];
#rmse_bcm2M = []; 
#rmse_gpoe2M = [];
#rmse_poe2M = [];
#rmse_sor2M = []; 
#
#add_nlpd_gbcm2M = [];
#add_nlpd_bcm2M = [];
#add_nlpd_gpoe2M = []; 
#add_nlpd_poe2M = []; 
#add_nlpd_sor2M = []; 
#
#
## 5M prelim. results 
#add_rmse_gbcm5M = []; 
#add_rmse_bcm5M = []; 
#add_rmse_gpoe5M = []; 
#add_rmse_poe5M = [];
#add_rmse_sor5M = []; 
#
#add_nlpd_gbcm5M = [];
#add_nlpd_bcm5M = [];
#add_nlpd_gpoe5M = [];
#add_nlpd_poe5M = [];
#add_nlpd_sor5M = []; 
 
for f in onlyfiles:
  with open(mypath+f, 'r') as ff:
    results = pk.load(ff)
    
    # get indices of test y-values
    i_s = results['i_s']    
    ytest = np.asarray(data.y)[i_s].squeeze()
    ytest = np.reshape(ytest, [len(ytest),1])
    
    
    # load the predictions
    ygbcm = np.asarray(results['ygbcm']).reshape(-1,1)
    sgbcm = np.asarray(results['sgbcm']).reshape(-1,1)
    ybcm = np.asarray(results['ybcm']).reshape(-1,1)
    sbcm = np.asarray(results['sbcm']).reshape(-1,1)
    ygpoe = np.asarray(results['ygpoe']).reshape(-1,1)
    sgpoe = np.asarray(results['sgpoe']).reshape(-1,1)
    try:
      ypoe = np.asarray(results['ypoe']).reshape(-1,1)
      spoe = np.asarray(results['spoe']).reshape(-1,1)
    except:
      ypoe = []; spoe = []; pass
    try:   
      ysor = np.asarray(results['ysor']).reshape(-1,1)
      ssor = np.asarray(results['ssor']).reshape(-1,1)
    except:
      ysor = []; ssor = []; pass

    # compute some statistics 
    if len(results['i_t']) == 700000:
      # cycle through minibatches
      for j in range(np.int(np.ceil(100000/batch_size))):
        batch_idx = range(j*batch_size, (j+1)*batch_size)
        
        if len(ygbcm)>0:  # check whether predictions are empty (experiment failed)
          #rmse
          se_gbcm700K += ((ytest[batch_idx]-ygbcm[batch_idx])**2,)
          #nlpd
          nll_gbcm700K += (nlpd(ygbcm[batch_idx], sgbcm[batch_idx], ytest[batch_idx]),)
        if len(ybcm)>0:
          #rmse
          se_bcm700K += ((ytest[batch_idx]-ybcm[batch_idx])**2,)
          #nlpd
          nll_bcm700K += (nlpd(ybcm[batch_idx], sbcm[batch_idx], ytest[batch_idx]),)
        if len(ygpoe)>0:
          #rmse
          se_gpoe700K += ((ytest[batch_idx]-ygpoe[batch_idx])**2,)
          #nlpd
          nll_gpoe700K += (nlpd(ygpoe[batch_idx], sgpoe[batch_idx], ytest[batch_idx]),)
        if len(ypoe)>0:
          #rmse
          se_poe700K += ((ytest[batch_idx]-ypoe[batch_idx])**2,)
          #nlpd
          nll_poe700K += (nlpd(ypoe[batch_idx], spoe[batch_idx], ytest[batch_idx]),)
        if len(ysor)>0:
          se_sor700K += ((ytest[batch_idx]-ysor[batch_idx])**2,)
          nll_sor700K += (nlpd(ysor[batch_idx], ssor[batch_idx], ytest[batch_idx]),)
      # RMSE for one file
      rmse_gbcm700K += (np.sqrt(np.mean(se_gbcm700K)),)
      rmse_bcm700K += (np.sqrt(np.mean(se_bcm700K)),)
      rmse_gpoe700K += (np.sqrt(np.mean(se_gpoe700K)),)
      rmse_poe700K += (np.sqrt(np.mean(se_poe700K)),)
      rmse_sor700K += (np.sqrt(np.mean(se_sor700K)),)
      
      nlpd_gbcm700K += (np.mean(nll_gbcm700K),)
      nlpd_bcm700K += (np.mean(nll_bcm700K),)
      nlpd_gpoe700K += (np.mean(nll_gpoe700K),)
      nlpd_poe700K += (np.mean(nll_poe700K),)
      nlpd_sor700K += (np.mean(nll_sor700K),)
      
      
      
            
      
        # figure out the architecture and numCPUs
       
      
    if len(results['i_t']) == 2000000 and results['numExperts']==8192:
      # cycle through minibatches
      for j in range(np.int(np.ceil(100000/batch_size))):
        batch_idx = range(j*batch_size, (j+1)*batch_size)
        
        if len(ygbcm)>0:  # check whether predictions are empty (experiment failed)
          #rmse
          se_gbcm2M += ((ytest[batch_idx]-ygbcm[batch_idx])**2,)
          #nlpd
          nll_gbcm2M += (nlpd(ygbcm[batch_idx], sgbcm[batch_idx], ytest[batch_idx]),)
        if len(ybcm)>0:
          #rmse
          se_bcm2M += ((ytest[batch_idx]-ybcm[batch_idx])**2,)
          #nlpd
          nll_bcm2M += (nlpd(ybcm[batch_idx], sbcm[batch_idx], ytest[batch_idx]),)
        if len(ygpoe)>0:
          #rmse
          se_gpoe2M += ((ytest[batch_idx]-ygpoe[batch_idx])**2,)
          #nlpd
          nll_gpoe2M += (nlpd(ygpoe[batch_idx], sgpoe[batch_idx], ytest[batch_idx]),)
        if len(ypoe)>0:
          #rmse
          se_poe2M += ((ytest[batch_idx]-ypoe[batch_idx])**2,)
          #nlpd
          nll_poe2M += (nlpd(ypoe[batch_idx], spoe[batch_idx], ytest[batch_idx]),)
        if len(ysor)>0:
          se_sor2M += ((ytest[batch_idx]-ysor[batch_idx])**2,)
          nll_sor2M += (nlpd(ysor[batch_idx], ssor[batch_idx], ytest[batch_idx]),)
          
       
      # RMSE and NLPD for a single file
      rmse_gbcm2M += (np.sqrt(np.mean(se_gbcm2M)),)
      rmse_bcm2M += (np.sqrt(np.mean(se_bcm2M)),)
      rmse_gpoe2M += (np.sqrt(np.mean(se_gpoe2M)),)
      rmse_poe2M += (np.sqrt(np.mean(se_poe2M)),)
      rmse_sor2M += (np.sqrt(np.mean(se_sor2M)),)   
      nlpd_gbcm2M += (np.mean(nll_gbcm2M),)
      nlpd_bcm2M += (np.mean(nll_bcm2M),)
      nlpd_gpoe2M += (np.mean(nll_gpoe2M),)
      nlpd_poe2M += (np.mean(nll_poe2M),)
      nlpd_sor2M += (np.mean(nll_sor2M),)
      
    if len(results['i_t']) == 5000000:
      # cycle through minibatches
      for j in range(np.int(np.ceil(100000/batch_size))):
        batch_idx = range(j*batch_size, (j+1)*batch_size)
        
        if len(ygbcm)>0:  # check whether predictions are empty (experiment failed)
          #rmse
          se_gbcm5M += ((ytest[batch_idx]-ygbcm[batch_idx])**2,)
          #nlpd
          nll_gbcm5M += (nlpd(ygbcm[batch_idx], sgbcm[batch_idx], ytest[batch_idx]),)
        if len(ybcm)>0:
          #rmse
          se_bcm5M += ((ytest[batch_idx]-ybcm[batch_idx])**2,)
          #nlpd
          nll_bcm5M += (nlpd(ybcm[batch_idx], sbcm[batch_idx], ytest[batch_idx]),)
        if len(ygpoe)>0:
          #rmse
          se_gpoe5M += ((ytest[batch_idx]-ygpoe[batch_idx])**2,)
          #nlpd
          nll_gpoe5M += (nlpd(ygpoe[batch_idx], sgpoe[batch_idx], ytest[batch_idx]),)
        if len(ypoe)>0:
          #rmse
          se_poe5M += ((ytest[batch_idx]-ypoe[batch_idx])**2,)
          #nlpd
          nll_poe5M += (nlpd(ypoe[batch_idx], spoe[batch_idx], ytest[batch_idx]),)
        if len(ysor)>0:
          #rmse
          se_sor5M += ((ytest[batch_idx]-ysor[batch_idx])**2,)
          #nlpd
          nll_sor5M += (nlpd(ysor[batch_idx], ssor[batch_idx], ytest[batch_idx]),)
        
     
      # RMSE and NLPD for one file
      rmse_gbcm5M += (np.sqrt(np.mean(se_gbcm5M)),)
      rmse_bcm5M += (np.sqrt(np.mean(se_bcm5M)),)
      rmse_gpoe5M += (np.sqrt(np.mean(se_gpoe5M)),)
      rmse_poe5M += (np.sqrt(np.mean(se_poe5M)),)      
      rmse_sor5M += (np.sqrt(np.mean(se_sor5M)),)      
      nlpd_gbcm5M += (np.mean(nll_gbcm5M),)
      nlpd_bcm5M += (np.mean(nll_bcm5M),)
      nlpd_gpoe5M += (np.mean(nll_gpoe5M),)
      nlpd_poe5M += (np.mean(nll_poe5M),)
      nlpd_sor5M += (np.mean(nll_sor5M),)
#      
#    if len(results['i_t']) == 5000000:
 
# add the preliminary results here....

rmse_gbcm700K+=add_rmse_gbcm700K
rmse_bcm700K+=add_rmse_bcm700K
rmse_gpoe700K+=add_rmse_gpoe700K 
rmse_poe700K+=add_rmse_poe700K

rmse_gbcm2M+=add_rmse_gbcm2M
rmse_bcm2M+=add_rmse_bcm2M
rmse_gpoe2M+=add_rmse_gpoe2M 
rmse_poe2M+=add_rmse_poe2M

rmse_gbcm5M+=add_rmse_gbcm5M
rmse_bcm5M+=add_rmse_bcm5M
rmse_gpoe5M+=add_rmse_gpoe5M 
rmse_poe5M+=add_rmse_poe5M

nlpd_gbcm700K+=add_nlpd_gbcm700K
nlpd_bcm700K+=add_nlpd_bcm700K
nlpd_gpoe700K+=add_nlpd_gpoe700K 
nlpd_poe700K+=add_nlpd_poe700K

nlpd_gbcm2M+=add_nlpd_gbcm2M
nlpd_bcm2M+=add_nlpd_bcm2M
nlpd_gpoe2M+=add_nlpd_gpoe2M 
nlpd_poe2M+=add_nlpd_poe2M

nlpd_gbcm5M+=add_nlpd_gbcm5M
nlpd_bcm5M+=add_nlpd_bcm5M
nlpd_gpoe5M+=add_nlpd_gpoe5M 
nlpd_poe5M+=add_nlpd_poe5M
    
print 'RMSE (700K)'
print 'rBCM:', np.mean(rmse_gbcm700K), '+/-', np.std(rmse_gbcm700K)/np.sqrt(len(rmse_gbcm700K))
print 'BCM :', np.mean(rmse_bcm700K), '+/-', np.std(rmse_bcm700K)/np.sqrt(len(rmse_bcm700K))
print 'gPoE:', np.mean(rmse_gpoe700K), '+/-', np.std(rmse_gpoe700K)/np.sqrt(len(rmse_gpoe700K))
print 'PoE :', np.mean(rmse_poe700K), '+/-', np.std(rmse_poe700K)/np.sqrt(len(rmse_poe700K))
print 'SOR :', np.mean(rmse_sor700K), '+/-', np.std(rmse_sor700K)/np.sqrt(len(rmse_sor700K))
  
print 'NLPD (700K)'
print 'rBCM:', np.mean(nlpd_gbcm700K), '+/-', np.std(nlpd_gbcm700K)/np.sqrt(len(nlpd_gbcm700K))
print 'BCM :', np.mean(nlpd_bcm700K), '+/-', np.std(nlpd_bcm700K)/np.sqrt(len(nlpd_bcm700K))
print 'gPoE:', np.mean(nlpd_gpoe700K), '+/-', np.std(nlpd_gpoe700K)/np.sqrt(len(nlpd_gpoe700K))
print 'PoE :', np.mean(nlpd_poe700K), '+/-', np.std(nlpd_poe700K)/np.sqrt(len(nlpd_poe700K))
print 'SOR :', np.mean(nlpd_sor700K), '+/-', np.std(nlpd_sor700K)/np.sqrt(len(nlpd_sor700K))  
  
print 'RMSE (2M)'
print 'rBCM:', np.mean(rmse_gbcm2M), '+/-', np.std(rmse_gbcm2M)/np.sqrt(len(rmse_gbcm2M))
print 'BCM :', np.mean(rmse_bcm2M), '+/-', np.std(rmse_bcm2M)/np.sqrt(len(rmse_bcm2M))
print 'gPoE:', np.mean(rmse_gpoe2M), '+/-', np.std(rmse_gpoe2M)/np.sqrt(len(rmse_gpoe2M))
print 'PoE :', np.mean(rmse_poe2M), '+/-', np.std(rmse_poe2M)/np.sqrt(len(rmse_poe2M))
print 'SOR :', np.mean(rmse_sor2M), '+/-', np.std(rmse_sor2M)/np.sqrt(len(rmse_sor2M))  
  
  
print 'NLPD (2M)'
print 'rBCM:', np.mean(nlpd_gbcm2M), '+/-', np.std(nlpd_gbcm2M)/np.sqrt(len(nlpd_gbcm2M))
print 'BCM :', np.mean(nlpd_bcm2M), '+/-', np.std(nlpd_bcm2M)/np.sqrt(len(nlpd_bcm2M))
print 'gPoE:', np.mean(nlpd_gpoe2M), '+/-', np.std(nlpd_gpoe2M)/np.sqrt(len(nlpd_gpoe2M))
print 'PoE :', np.mean(nlpd_poe2M), '+/-', np.std(nlpd_poe2M)/np.sqrt(len(nlpd_poe2M))
print 'SOR :', np.mean(nlpd_sor2M), '+/-', np.std(nlpd_sor2M)/np.sqrt(len(nlpd_sor2M))
  
print 'RMSE (5M)'
print 'rBCM:', np.mean(rmse_gbcm5M), '+/-', np.std(rmse_gbcm5M)/np.sqrt(len(rmse_gbcm5M))
print 'BCM :', np.mean(rmse_bcm5M), '+/-', np.std(rmse_bcm5M)/np.sqrt(len(rmse_bcm5M))
print 'gPoE:', np.mean(rmse_gpoe5M), '+/-', np.std(rmse_gpoe5M)/np.sqrt(len(rmse_gpoe5M))
print 'PoE :', np.mean(rmse_poe5M), '+/-', np.std(rmse_poe5M)/np.sqrt(len(rmse_poe5M))
print 'SOR :', np.mean(rmse_sor5M), '+/-', np.std(rmse_sor5M)/np.sqrt(len(rmse_sor5M)) 

 
print 'NLPD (5M)'
print 'rBCM:', np.mean(nlpd_gbcm5M), '+/-', np.std(nlpd_gbcm5M)/np.sqrt(len(nlpd_gbcm5M))
print 'BCM :', np.mean(nlpd_bcm5M), '+/-', np.std(nlpd_bcm5M)/np.sqrt(len(nlpd_bcm5M))
print 'gPoE:', np.mean(nlpd_gpoe5M), '+/-', np.std(nlpd_gpoe5M)/np.sqrt(len(nlpd_gpoe5M))
print 'PoE :', np.mean(nlpd_poe5M), '+/-', np.std(nlpd_poe5M)/np.sqrt(len(nlpd_poe5M))
print 'SOR :', np.mean(nlpd_sor5M), '+/-', np.std(nlpd_sor5M)/np.sqrt(len(nlpd_sor5M))