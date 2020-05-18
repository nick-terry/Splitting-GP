# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:09:11 2015

@author: marc
"""

import sys
import numpy as np
from random import shuffle
from random import seed
#import matplotlib.pyplot as plt
import cPickle as pk
import string
from hgp import HGP
from hgp.utils.tictoc import *
from hgp.utils.analyze import *
from multiprocessing import cpu_count
from hgp import GP
from hgp import gBCM
from hgp import gPoE
import datetime
import time
import matplotlib.pyplot as plt

def kin40k_experiment(randomSeed=1, numExperts=512):

  
#  try:
#      randomSeed = int(sys.argv[0])
#      #arch = [ int(x) for x in string.split(sys.argv[1],',') ]
#  except:
#      randomSeed = 2;
#      pass
#  try:
#      numExperts = int(sys.argv[1])
#      #arch = [ int(x) for x in string.split(sys.argv[1],',') ]
#  except:
#      numExperts = 256;
#      pass
  
  np.random.seed(randomSeed)
  
  print 'randomSeed', str(randomSeed)
  print '#Experts', str(numExperts)
  
  numCPUs = cpu_count()
  
  
  
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
  
  print 'Load data...'
  Xtrain = np.asarray(getData('./data/kin40K/kin40k_train_data.asc'))
  ytrain = np.asarray(getData('./data/kin40K/kin40k_train_labels.asc')).squeeze()
  Xtest = np.asarray(getData('./data/kin40K/kin40k_test_data.asc'))
  ytest = np.asarray(getData('./data/kin40K/kin40k_test_labels.asc')).squeeze()
  
  hypers = np.asarray(getData('./experiments/kin40params.txt')).squeeze()
  
  train_N = len(ytrain)
  test_N = len(ytest)
  
  results = {}
  ############################
  ## GP training
  gp = GP(Xtrain, ytrain, cov_type='covSEard')
  #tic()
  #gp.train()
  #gp_trainTime = toc()
  #print gp.params
  gp.params = hypers;
  
  # time for gradient of NLML
  #tic();
  #gp.NLML(derivs=True)
  #gptime = toc();
  #results['gp-NLML_gradient-time'] = gptime
  
  ## GP predictions
  print 'Predicting..'
  batch_size = 1000
  
  if False:
      print 'Train Full GP'
      ymu = []
      ys2 = []
      smse = 0
      NLPD = []
      for i in range(np.int(np.ceil(Xtest.shape[0]/batch_size))):
          batch_idx = range(i*batch_size, (i+1)*batch_size)
          m,s = gp.predict(Xtest[batch_idx,:], variance=True)
          ymu += (m,)
          ys2 += (s,)
          print (i+1),'/',np.int(np.ceil(Xtest.shape[0]/batch_size))
          smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          NLPD += (nlpd(m, s, ytest[batch_idx]),)
  
      meanNLPD = np.mean(NLPD)
      print 'SMSE: %f' % smse
      print 'NLPD: %f' % meanNLPD
  
      results = {}
      #results['bfgs_niter'] = hgp1.lik_count
      results['train_time'] = time
      results['gp_params'] = gp.params
      results['smse'] = smse
      results['ytest'] = ytest
      results['ygp'] = ymu
      results['sgp'] = ys2
      results['gpNLPD'] = NLPD
  
      st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H-%M-%S')
      filename = 'results/'+st+'_results_kin40K_GPonly_'+str(numCPUs)+'CPUs.pk'
      with open(filename,'wb') as f:
          pk.dump(results,f)
      
      print 'results saved in '+filename
  
  
  ############################ gBCM experiment
  
  if True:
      numDataPerExpert = train_N/numExperts
      print 'gBCM'
      # select some architecture candidates
      print 'Finding optimal architecture...'
      branchingFactor = []
      depth = []
      for i in range(50):
          k = numCPUs**i
          if k==numExperts:
              branchingFactor += (numCPUs,)
              depth += (i,)
          l = (numCPUs/2)**i
          if l==numExperts:
              branchingFactor += (numCPUs/2,)
              depth += (i,)
          k = (numCPUs/4)**i
          if k==numExperts:
              branchingFactor += (numCPUs/4,)
              depth += (i,)
      print 'Architecture options:'
      print branchingFactor
      print depth
      # find the best architecture
      nlml = []
      TT = []
      for i in range(len(branchingFactor)):
          profile = [(branchingFactor[i],'random','rep1')]*depth[i]
          try:
              gbcm = gBCM.gBCM(Xtrain, ytrain, profile=profile)
              tic()
              gbcm.NLML(derivs=True)
              TT += (toc(),)
              nlml += (gbcm.NLML(),)
              del gbcm
          except:
              nlml += (np.Inf,)
              TT += (np.Inf,)
              pass
      print str(TT)
      print str(nlml)
      utility =  np.asarray(TT)*np.asarray(nlml)
      optArchIdx = np.argmin(utility)
      branchingFactor = branchingFactor[optArchIdx]
      depth = depth[optArchIdx]
  
      numExperts = branchingFactor**depth
      numDataPerExpert = train_N/numExperts
  
      results['NLML_gradient-time'] = TT[optArchIdx]
      results['numCPUs'] = numCPUs
      results['branchingFactor'] = branchingFactor
      results['depth'] = depth
      results['numExperts'] = numExperts
      results['numDataPerExpert'] =  numDataPerExpert
  
  
      print '---------------------'
      print 'Optimal architecture:'
      print 'Branching Factor: ', branchingFactor
      print 'Depth: ', depth
      print '#CPUs: ', numCPUs
      print '#Experts: ', numExperts
      print '#Points/Expert: ', train_N/numExperts
      print '---------------------'
  
      # set up the model
      profile = [(branchingFactor,'random','rep1')]*depth
      #profile = [(numExperts,'random','rep1')]*1
      gbcm = gBCM.gBCM(Xtrain,ytrain,profile=profile, pool='default')
      gbcm.params = hypers
  #    poe = HGP(Xtrain, ytrain, profile=profile, pool = 'default')
  #    poe.params=hypers
  #    gpoe_profile = [(numExperts,'random','rep1')]*1
  #    gpoe = gPoE.gPoE(Xtrain,ytrain,profile=gpoe_profile, pool='default')
  #    gpoe.params = hypers
      # gBCM
      ygbcm = []
      sgbcm = []
      gbcm_smse = 0
      gbcmNLPD = []
      gbcm_se = []
      # gPoE
      ygpoe = []
      sgpoe = []
      gpoe_smse = 0
      gpoeNLPD = []
      gpoe_se = []
      # BCM
      ybcm = []
      sbcm = []
      bcm_smse = 0
      bcmNLPD = []
      bcm_se = []
      # PoE
      ypoe = []
      spoe = []
      poe_smse = 0
      poeNLPD = []
      poe_se = []
      for i in range(np.int(np.ceil(Xtest.shape[0]/batch_size))):
          batch_idx = range(i*batch_size, (i+1)*batch_size)
          print '-----------------------------------------------------'
          print (i+1),'/',np.int(np.ceil(Xtest.shape[0]/batch_size))
          
          
          # gBCM prediction
          if hasattr(gbcm, 'beta'):
              del gbcm.beta
          gbcm.correction = True
          m,s = gbcm.predict(Xtest[batch_idx,:], latent_variance=True)
          s += np.exp(gbcm.params[-1])
          ygbcm += (m,)
          sgbcm += (s,)
          gbcm_se += ((m-ytest[batch_idx])**2,)
          gbcm_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          gbcmNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          # gPoE prediction
          gbcm.beta=1.0/gbcm.nleaf
          gbcm.correction = False
          m,s = gbcm.predict(Xtest[batch_idx,:], latent_variance=True)
          s += np.exp(gbcm.params[-1])
          ygpoe += (m,)
          sgpoe += (s,)
          gpoe_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          gpoe_se += ((m-ytest[batch_idx])**2,)
          gpoeNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          # BCM prediction
          gbcm.beta=1.0
          gbcm.correction = True
          m,s = gbcm.predict(Xtest[batch_idx,:], latent_variance=True)
          s += np.exp(gbcm.params[-1])
          ybcm += (m,)
          sbcm += (s,)
          bcm_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          bcm_se += ((m-ytest[batch_idx])**2,)
          bcmNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          # PoE prediction: BCM without correction
          gbcm.beta=1.0
          gbcm.correction = False
          m,s = gbcm.predict(Xtest[batch_idx,:], latent_variance=True)
          del gbcm.correction
          s += np.exp(gbcm.params[-1])
          ypoe += (m,)
          spoe += (s,)
          poe_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          poe_se += ((m-ytest[batch_idx])**2,)
          poeNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
  
          print 'RMSE gBCM (%4.4e)' % np.sqrt(np.mean(gbcm_se))
          print 'RMSE  BCM (%4.4e)' % np.sqrt(np.mean(bcm_se))
          print 'RMSE gPoE (%4.4e)' % np.sqrt(np.mean(gpoe_se))
          print 'RMSE  PoE (%4.4e)' % np.sqrt(np.mean(poe_se))
          print ' '
          print 'NLPD gBCM (%4.4e)' % np.mean(np.asarray(gbcmNLPD))
          print 'NLPD  BCM (%4.4e)' % np.mean(np.asarray(bcmNLPD))
          print 'NLPD gPoE (%4.4e)' % np.mean(np.asarray(gpoeNLPD))
          print 'NLPD  PoE (%4.4e)' % np.mean(np.asarray(poeNLPD))
  
          #meanSMSE = np.mean(gbcm_smse)
          meanNLPD = np.mean(np.asarray(gbcmNLPD))
          #print 'SMSE: %f' % gbcm_smse
          #print 'NLPD: %f' % meanNLPD
          
      results['gbcm_params'] = gbcm.params
      results['ygbcm'] = ygbcm
      results['sgbcm'] = sgbcm
      #results['gbcm_smse'] = gbcm_smse
      #results['gbcmNLPD'] = gbcmNLPD
      #results['gbcm_se'] = gbcm_se
      results['ybcm'] = ybcm
      results['sbcm'] = sbcm
      #results['bcm_smse'] = bcm_smse
      #results['bcmNLPD'] = bcmNLPD
      #results['bcm_se'] = bcm_se
      results['ygpoe'] = ygpoe
      results['sgpoe'] = sgpoe
      #results['gpoe_smse'] = gpoe_smse
      #results['gpoeNLPD'] = gpoeNLPD
      #results['gpoe_se'] = gpoe_se
      results['ypoe'] = ypoe
      results['spoe'] = spoe
      #results['poe_smse'] = poe_smse
      #results['poeNLPD'] = poeNLPD
      #results['poe_se'] = poe_se
  
  ######################## SOR experiment
  if True:
      numExperiments = 1
      ysor = []
      ssor = []
      sor_smse = 0
      sorNLPD = []
      sor_se = []
  
      candidates = [100, 200, 300, 400, 500, 600, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
      TTSOR = []
      for i in candidates:
          sor_cand = int(i)
          ii = [[kk] for kk in range(Xtrain.shape[0])]
          shuffle(ii)
          idx = np.asarray(ii[0:sor_cand]).squeeze()
          XSOR = Xtrain[idx,:]
          ySOR = ytrain[idx]
          gpsor = GP(XSOR, ySOR)
          tic()
          gpsor.NLML(derivs=True)
          TTSOR += (toc(),)
          if TTSOR[-1] > TT[optArchIdx]: # found #SOR
              print '#regressors:',numSORs
              break
          numSORs = int(i)
  
  
      for i in range(numExperiments):
          # select random subset of the training data as SOR
          ii = [[kk] for kk in range(Xtrain.shape[0])]
          shuffle(ii)
          idx = np.asarray(ii[0:numSORs]).squeeze()
          XSOR = Xtrain[idx,:]
          ySOR = ytrain[idx]
          gpsor = GP(XSOR, ySOR)
          # gpsor.train()
          # do the predictions
  
          ysor = []
          ssor = []
  
          for j in range(np.int(np.ceil(Xtest.shape[0]/batch_size))):
              print (j+1),'/',np.int(np.ceil(Xtest.shape[0]/batch_size))
              batch_idx = range(j*batch_size, (j+1)*batch_size)
              m,s = gpsor.predict(Xtest[batch_idx,:], variance=True)
              ysor += (m,)
              ssor += (s,)
  
              sor_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
              sor_se += ((m-ytest[batch_idx])**2,)
              sorNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
         # meanNLPDSOR = np.mean(sorNLPD)
          print 'RMSE SOR (%4.4e)' % np.sqrt(np.mean(sor_se))
          print 'NLPD SOR (%4.4e)' % np.mean(np.asarray(sorNLPD))
  
      results['ysor'] = ysor
      results['ssor'] = ssor
      results['numSORs'] = numSORs
#      results['sor_smse'] = sor_smse
#      results['sorNLPD'] = sorNLPD
#      results['sor_se'] = sor_se
      results['SOR_experiments'] = numExperiments
  
  st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H-%M-%S')
  filename = 'results/'+st+'_results_kin40K_'+str(numExperts)+'experts_'+str(numCPUs)+'CPUs_'+str(branchingFactor)+'_'+str(depth)+'_'+str(randomSeed)+'.pk'
  with open(filename,'wb') as f:
      pk.dump(results,f)
  
  print 'results saved in '+filename
