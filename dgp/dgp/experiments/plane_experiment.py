import sys

# from hgp.utils import getdata as gd
from hgp import DataSet as DS
from hgp import HGP
from hgp import GP
from hgp.utils.tictoc import *
import numpy as np
from random import shuffle
#import matplotlib.pyplot as plt
import cPickle as pk
import string
import math
from hgp import gBCM
from hgp.utils.analyze import *
import datetime
import time
from multiprocessing import cpu_count
import os, inspect
import pandas as pd

def plane_experiment(randomSeed=555, train_N = 2000000, test_N=100000):

  
  numCPUs = cpu_count()
  print '#CPUs:', numCPUs
  
  
  def usage(message=None):
      if message is not None:
          print '\n'+message
      print 'usage: python experiment.py <architecture> <training size> <test size>'
      print 'example: python experiment.py 4,4,4 1000000 100000\n'
      raise Exception
#  try:
#      randomSeed = int(sys.argv[0])
#      #arch = [ int(x) for x in string.split(sys.argv[1],',') ]
#  except:
#      randomSeed = 91;
#      pass
#  try:
#      train_N = int(sys.argv[1])
#  except:
#      train_N = 5000000
#      pass
#  try:
#      test_N = int(sys.argv[2])
#  except:
#      test_N =  100000
#      pass
#  
#  
  # fix the number of experts
  if train_N == 700000:
    numExperts = 4096
  if train_N == 2000000:
    numExperts = 8192#16384
  if train_N == 5000000:
    numExperts = 32768
    
  
  
  def removeNAN(x):
     y = np.asarray(x)
     idx = []
     for i in range(y.shape[0]):
         if math.isnan(y[i,:]):
             idx.append(i)
     return idx
  
  
  np.random.seed(randomSeed)
  
#  with open(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/../data/plane/filtered_data.pickle','r') as f:
#      data = pd.read_pickle(f)
  data = pd.read_pickle(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/../data/plane/filtered_data.pickle')
      #data = pk.load(f)
  
  y = data['ArrDelay']
  X = data[['Month','DayofMonth','DayOfWeek','DepTime','ArrTime','AirTime','Distance','plane_age']]
  
  # hgp0 = hgp.HGP(X,y,pool='default',profile=[(100,'random','rep2')]*3) # create GP with default cov
  
  data = DS(X,y)
  
  
  print 'Training Set size: ', train_N
  print 'Test Set size:     ', test_N
  print 'Number of Experts:', numExperts
  
  
  r = range(train_N+test_N)
  # r = range(1,len(y))
  shuffle(r)
  
  
  # i_t, i_s - training, test data indices from full data set
  i_t = r[:train_N]
  i_s = r[train_N:(train_N+test_N)]
  
  # Xt, yt - training data
  # Xs, ys - test data
  #Xt, yt = data.subset(i_t).data()
  #Xs, ys = data.subset(i_s).data()
  
  Xtrain = np.asarray(data.X)[i_t,:]
  ytrain = np.asarray(data.y)[i_t]
  
  XtrainWhitened, Xtrainmean, Xtrainstd = whitenData(Xtrain)
  ytrainWhitened, ytrainmean, ytrainstd = whitenData(ytrain)
  
  Xtest = np.asarray(data.X)[i_s,:]
  ytest = np.asarray(data.y)[i_s]
  
  XtestWhitened = whitenData(Xtest, Xtrainmean, Xtrainstd)
  ytestWhitened = whitenData(ytest, ytrainmean, ytrainstd)
  
  data = {}

  
  batch_size = 1000
  
  results = {}
  
  if True:
      numDataPerExpert = train_N/numExperts
      # select some architecture candidates
      print 'Finding optimal architecture...'
      branchingFactor = []
      depth = []
      for i in range(50):
        for kk in [64]: # [4,8,16,32,64,128,256,512,1024,2048,8192]:
          if kk**i==numExperts:
            branchingFactor += (kk,)
            depth += (i,)
  
      print 'Architecture options:'
      print branchingFactor
      print depth
      # find the best architecture
      nlml = []
      TT = []
      if True:
        for i in range(len(branchingFactor)):
            profile = [(branchingFactor[i],'random','rep1')]*depth[i]
            try:
                gbcm = gBCM.gBCM(XtrainWhitened, ytrainWhitened, profile=profile)
                tic()
                gbcm.NLML(derivs=True)
                TT += (toc(),)
                #nlml += (gbcm.NLML(),)
            except:
                nlml += (np.Inf,)
                TT += (np.Inf,)
                print 'Failed with architecture', branchingFactor[i], depth[i]
                pass
            gbcm = {}
        print str(TT)
        print str(nlml)
        #utility =  np.asarray(TT)*np.asarray(nlml)
        utility = np.asarray(TT)
        optArchIdx = np.argmin(utility)
        results['NLML_gradient-time'] = TT[optArchIdx]
      else:
        optArchIdx = 0
        
      branchingFactor = branchingFactor[optArchIdx]
      depth = depth[optArchIdx]
  
  
      numExperts = branchingFactor**depth
      numDataPerExpert = train_N/numExperts
  
      #results['NLML_gradient-time'] = TT[optArchIdx]
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
  
      #if numExperts==8192:
      #  profile = [(16,'random','rep1'),(16,'random','rep1'),(16,'random','rep1'),(2,'random','rep1')]
      #else:
      profile = [(branchingFactor,'random','rep1')]*depth    
        
      print 'Profile:' 
      print profile 
      gbcm = gBCM.gBCM(XtrainWhitened, ytrainWhitened, profile=profile)
  
  
      print 'Training..'
      tic()
      gbcm.train()
      trainTime = toc()
  
      # copy the parameters over
  #    bcm.params=gbcm.params
  #    gpoe.params=gbcm.params
  
  
  
      ## Predictions
      print 'Predicting..'
  
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
      # HGP
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
          m,s = gbcm.predict(XtestWhitened[batch_idx,:], latent_variance=True)
          s += np.exp(gbcm.params[-1])
          m,s = unWhitenPrediction(m,s,ytrainmean,ytrainstd)
          ygbcm += (m,)
          sgbcm += (s,)
          gbcm_se += ((m-ytest[batch_idx])**2,)
          gbcm_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          gbcmNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          # gPoE prediction
          gbcm.beta=1.0/gbcm.nleaf
          gbcm.correction = False
          m,s = gbcm.predict(XtestWhitened[batch_idx,:], latent_variance=True)
          s += np.exp(gbcm.params[-1])
          m,s = unWhitenPrediction(m,s,ytrainmean,ytrainstd)
          ygpoe += (m,)
          sgpoe += (s,)
          gpoe_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          gpoe_se += ((m-ytest[batch_idx])**2,)
          gpoeNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          # BCM prediction
          gbcm.beta=1.0
          gbcm.correction = True
          m,s = gbcm.predict(XtestWhitened[batch_idx,:], latent_variance=True)
          s += np.exp(gbcm.params[-1])
          m,s = unWhitenPrediction(m,s,ytrainmean,ytrainstd)
          ybcm += (m,)
          sbcm += (s,)
          bcm_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
          bcm_se += ((m-ytest[batch_idx])**2,)
          bcmNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          # PoE prediction: BCM without correction
          gbcm.beta=1.0
          gbcm.correction = False
          m,s = gbcm.predict(XtestWhitened[batch_idx,:], latent_variance=True)
          del gbcm.correction
          s += np.exp(gbcm.params[-1])
          m,s = unWhitenPrediction(m,s,ytrainmean,ytrainstd)
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
  
  
      # meanNLPD = np.mean(np.asarray(gbcmNLPD))
      # print 'SMSE: %f' % gbcm_smse
      # print 'NLPD: %f' % meanNLPD
  
      results['models'] = ['gBCM', 'gPoE', 'BCM', 'PoE']
      results['train_time'] = trainTime
      results['gbcm_params'] = gbcm.params
      results['ygbcm'] = ygbcm
      results['sgbcm'] = sgbcm
  #    results['gbcm_smse'] = gbcm_smse
  #    results['gbcmNLPD'] = gbcmNLPD
  #    results['gbcm_se'] = gbcm_se
      results['ybcm'] = ybcm
      results['sbcm'] = sbcm
  #    results['bcm_smse'] = bcm_smse
  #    results['bcmNLPD'] = bcmNLPD
  #    results['bcm_se'] = bcm_se
      results['ygpoe'] = ygpoe
      results['sgpoe'] = sgpoe
  #    results['gpoe_smse'] = gpoe_smse
  #    results['gpoeNLPD'] = gpoeNLPD
  #    results['gpoe_se'] = gpoe_se
      results['ypoe'] = ypoe
      results['spoe'] = spoe
  
      results['i_t'] = i_t
      results['i_s'] = i_s
      results['architecture_bf'] = branchingFactor
      results['architecture_depth'] = depth
  
  #    # free some memory
  #    params = gbcm.params
  ##    del gbcm
  ##    poe = HGP(XtrainWhitened, ytrainWhitened, profile=profile)
  ##    poe.params=params
  #    for i in range(np.int(np.ceil(Xtest.shape[0]/batch_size))):
  #        batch_idx = range(i*batch_size, (i+1)*batch_size)
  #
  #        print '-----------------------------------------------------'
  #        print (i+1),'/',np.int(np.ceil(Xtest.shape[0]/batch_size))
  #        # PoE prediction
  #        m,s = poe.predict(XtestWhitened[batch_idx,:], latent_variance=True)
  #        s += np.exp(poe.params[-1])
  #        m, s = unWhitenPrediction(m,s,ytrainmean,ytrainstd)
  #        #######
  #        ypoe += (m,)
  #        spoe += (s,)
  #        poe_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
  #        poe_se += ((m-ytest[batch_idx])**2,)
  #        poeNLPD += (nlpd(m, s, ytest[batch_idx]),)
  #        print 'RMSE  PoE (%4.4e)' % np.sqrt(np.mean(poe_se))
  #        print 'NLPD  PoE (%4.4e)' % np.mean(np.asarray(poeNLPD))
  #
  #    results['ypoe'] = ypoe
  #    results['spoe'] = spoe
  #    results['poe_smse'] = poe_smse
  #    results['poeNLPD'] = poeNLPD
  #    results['poe_se'] = poe_se
  
  ######################## SOR experiment
  if True:
      numExperiments = 1
      ysor = []
      ssor = []
      #sor_smse = 0
      sorNLPD = []
      sor_se = []
  
      candidates = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
      TTSOR = []
      for i in candidates:
          numSORs = int(i)
          ii = [[kk] for kk in range(Xtrain.shape[0])]
          shuffle(ii)
          idx = np.asarray(ii[0:numSORs]).squeeze()
          XSOR = XtrainWhitened[idx,:]
          ySOR = ytrainWhitened[idx]
          gpsor = GP(XSOR, ySOR)
          tic()
          gpsor.NLML(derivs=True)
          TTSOR += (toc(),)
          if TTSOR[-1] > TT[optArchIdx]: # found #SOR
              print '#regressors:',i
              break
  
  
  
      for i in range(numExperiments):
          # select random subset of the training data as SOR
          ii = [[kk] for kk in range(Xtrain.shape[0])]
          shuffle(ii)
          idx = np.asarray(ii[0:numSORs]).squeeze()
          XSOR = Xtrain[idx,:]
          ySOR = ytrain[idx]
          gpsor = GP(XSOR, ySOR)
          gpsor.train()
          # do the predictions
  
          ysor = []
          ssor = []
  
          for j in range(np.int(np.ceil(Xtest.shape[0]/batch_size))):
              print (j+1),'/',np.int(np.ceil(Xtest.shape[0]/batch_size))
              batch_idx = range(j*batch_size, (j+1)*batch_size)
              m,s = gpsor.predict(XtestWhitened[batch_idx,:], variance=True)
              m,s = unWhitenPrediction(m,s,ytrainmean,ytrainstd)
              ysor += (m,)
              ssor += (s,)
  
              #sor_smse += np.mean((m-ytest[batch_idx])**2)/np.var(ytest)
              sor_se += ((m-ytest[batch_idx])**2,)
              sorNLPD += (nlpd(m, s, ytest[batch_idx]),)
  
          meanNLPDSOR = np.mean(sorNLPD)
          print 'RMSE SOR (%4.4e)' % np.sqrt(np.mean(sor_se))
          print 'NLPD SOR (%4.4e)' % np.mean(np.asarray(sorNLPD))
  
      results['ysor'] = ysor
      results['ssor'] = ssor
      #results['sor_smse'] = sor_smse
      #results['sorNLPD'] = sorNLPD
      #results['sor_se'] = sor_se
      results['SOR_experiments'] = numExperiments
      results['numSORs'] = numSORs
  
  
  
  st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
  filename = st+'_results_plane_'+str(train_N)+'_'+str(branchingFactor)+'_'+str(depth)+'_'+str(randomSeed)+'.pk'
  with open(filename,'wb') as f:
      pk.dump(results,f)
  
  print 'results saved in '+filename


