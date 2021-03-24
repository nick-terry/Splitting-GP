# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from extra_kernels import EpanechnikovKernel
        
        
        

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   

def getData():
    
    n=501
    x = torch.linspace(-100,200,steps=n)
    f1 = torch.cat([5*torch.sin(-3+.2*x[:ceil(.23*n)]),
                        0*torch.sin(.1*x[ceil(.23*n):ceil(.85*n)]),
                        5*torch.sin(2.8+.2*x[ceil(.85*n):])])
    nDist1,nDist2 = torch.distributions.Normal(110, 20),torch.distributions.Normal(-10, 20)  
    f2 = 50*torch.exp(nDist1.log_prob(x)) + 100*torch.exp(nDist2.log_prob(x))
    sigma2 = .5
    
    x = x - torch.mean(x)
    x = x/torch.std(x)
    f1 = f1 - torch.mean(f1)
    f1 = f1/torch.std(f1)
    
    # true function
    y = f1 + torch.sqrt(sigma2 * torch.exp(f2)) * torch.randn(x.size()[0])#continue here
    ytrue = f1[::2]
    xtrue = x[::2]
    ntrue = xtrue.size()[0]
    
    xtrain = x[::2]
    ytrain = y[::2]
    ntrain = xtrain.size()[0]
    
    xtest = x[1::4]
    ytest = y[1::4]
    ntest = xtest.size()[0]
    
    # get a validation and test set
    # dataset = torch.utils.data.TensorDataset(xte,yte)
    # validation,test = torch.utils.data.random.split(dataset,
    #                                                 lengths=[nte//2,nte-nte//2],
    #                                                 generator=torch.Generator().manual_seed(42))
    
    return x,y,xtrain,ytrain,xtest,ytest,xtrue,ytrue

def getData2():
    
    n=501
    x = torch.linspace(-100,200,steps=n)
    # f1 = torch.cat([5*torch.sin(-3+.2*x[:ceil(.23*n)]),
    #                    0*torch.sin(.1*x[ceil(.23*n):ceil(.85*n)]),
    #                    5*torch.sin(2.8+.2*x[ceil(.85*n):])])
    f1 = (x**2)/200**2 - (x**3)/(200**3)/2
    nDist1,nDist2 = torch.distributions.Normal(0, .05),torch.distributions.Normal(0, .02)  
    
    x = x - torch.mean(x)
    x = x/torch.std(x)
    f1 = f1 - torch.mean(f1)
    f1 = f1/torch.std(f1)
    
    # true function
    eta1 = nDist1.sample((n//2,))
    eta2 = nDist2.sample((n//2,))
    eta = torch.cat((eta1,torch.zeros(n-2*(n//2)),eta2))
    y = f1 + eta
    ytrue = f1[::2]
    xtrue = x[::2]
    ntrue = xtrue.size()[0]
    
    xtrain = x[::2]
    ytrain = y[::2]
    ntrain = xtrain.size()[0]
    
    xtest = x[1::4]
    ytest = y[1::4]
    ntest = xtest.size()[0]
    
    # get a validation and test set
    # dataset = torch.utils.data.TensorDataset(xte,yte)
    # validation,test = torch.utils.data.random.split(dataset,
    #                                                 lengths=[nte//2,nte-nte//2],
    #                                                 generator=torch.Generator().manual_seed(42))
    
    return x,y,xtrain,ytrain,xtest,ytest,xtrue,ytrue

def split(X,splitPoint):
    labels = X<splitPoint
    return X[labels],X[~labels],labels

#Fix the random seeds for using CRN with rBCM experiment
torch.manual_seed(10101)
np.random.seed(10101)

x,y,xtrain,ytrain,xtest,ytest,xtrue,ytrue = getData()    

def trainGP(x,y,xtrue,ytrue,nEpochs=3,trueLoss=False):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood)
    
    model.train()
    likelihood.train()
    
    # set up mll and optimizer
    opt = torch.optim.Adam(model.parameters(), lr=.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = torch.zeros((nEpochs,1))
    trueLosses = torch.zeros((nEpochs,1))
    
    #Up this setting to prevent potential numerical issues if CG hasn't converged in <2000 iterations
    for epoch in range(nEpochs):
        
        opt.zero_grad()
        output = model(x)
        
        loss = -mll(output,y)
        loss.backward()
        
        mse = torch.mean((output.mean-y)**2)
        losses[epoch] = mse.detach()
        
        if trueLoss:
            tmse = torch.mean((model(xtrue).mean-ytrue)**2)
            trueLosses[epoch] = tmse.detach()
        
        print('Epoch {} has loss: {}'.format(epoch,loss))
        
        opt.step()
    
    model.eval()
    
    return model,losses,trueLosses

model,losses,trueLosses = trainGP(xtrain,ytrain,xtrue,ytrue,trueLoss=True,nEpochs=1000)

fig,ax = plt.subplots(1)
ax.plot(losses,label='Training Loss',alpha=.5)
ax.plot(trueLosses,label='True Loss (w/ noiseless response)',alpha=.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Regular GP Losses')
ax.legend()

f_pred = model(xtrue)

# compute some gradients
# nSteps = 10
# gradGrid = torch.linspace(-1,1,steps=nSteps)
# trueGrads = trueGrad(gradGrid)
# trueGradsSc = trueGrads/torch.norm(trueGrads,p=2,dim=1,keepdim=True)/25
# estGrads = estGrad(gradGrid, model).detach()
# estGradsSc = estGrads/torch.norm(estGrads,p=2,dim=1,keepdim=True)/25

# compute MSE of grad vectors
mse = torch.mean((f_pred.mean-ytrue)**2)
# gradMse = torch.mean(torch.norm(trueGrads-estGrads,p=2,dim=1),dim=0)

# try using eigenvector of C matrix to split
# C = getC(estGrads)
# eigenvalues are sorted in ascending order, so take the last eigenvector to split
# Lambda,eigV = torch.symeig(C,eigenvectors=True)
# center = torch.mean(X_train,dim=0)

def getLatentProbs(x,xList,models):
    
    parent,children = models[0],models[1:]
    
    PArr = torch.zeros((len(children),x.shape[0]))
    # compute covar and sum over all data assigned to the child model
    for i,model in enumerate(children):
        PArr[i] = torch.sum(parent.covar_module(x,xList[i]).evaluate(),dim=1)
    
    # normalize the summed covar to get KDE
    PArr = PArr / torch.sum(PArr,axis=0)
    
    return PArr

def getMllForDataAndModel(x,y,model):
    model.eval()
    output = model(x)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
    
    return mll(output,y)

def getMllForData(x,y,xModels,models):
    PArr = getLatentProbs(x, xModels, models)
    # approximate the latent probabilities of each child model
    Pz = torch.sum(PArr,dim=1)/torch.sum(PArr)
    
    # get mll of of data for each model
    children = models[1:]
    mllArr = torch.zeros((len(children),))
    for i,child in enumerate(children):
        mllArr[i] = getMllForDataAndModel(x, y, child)
    
    mll = torch.sum(Pz * torch.exp(mllArr))
    
    return mll

def getLogMll(xList,yList,xModels,models):
    
    mllArr = torch.zeros((len(xList),))
    for i in range(len(xList)):
        mllArr[i] = getMllForData(xList[i], yList[i], xModels, models)
        
    return torch.sum(torch.log(mllArr))

def splitPred(x,xList,models):
    
    parent,children = models[0],models[1:]
    
    yHatArr = torch.zeros((len(children),x.shape[0]))
    # # compute predictions with the new local models
    for i,child in enumerate(children):
        yHatArr[i] = child(x).mean

    PArr = getLatentProbs(x,xList,models)
    # sum over latent variable prob to get posterior mean
    yHat = torch.sum(PArr * yHatArr,axis=0)
    
    return yHat

def trainSplitGP(xList,yList,xModels,models,nEpochs=3):
    
    # set up mll and optimizer
    params = []
    for model in models:
        params += list(model.parameters())
    opt = torch.optim.Adam(params, lr=.1)
    
    losses = torch.zeros((nEpochs,))
    
    for epoch in range(nEpochs):
        
        opt.zero_grad()
        
        loss = -getLogMll(xList=(x_L,x_R),
                      yList=(y_L,y_R),
                      xModels=(x_L,x_R),
                      models=(model,model_L,model_R))
        loss.backward()
        
        print('Epoch {} has loss: {}'.format(epoch,loss))
        
        opt.step()
            
        losses[epoch] = loss.detach()
        
        
    return models,losses

def fitParams(xtr,ytr,xte,yte,xtrue,ytrue,xList,models,nEpochs=3):
    
    paramNames = [item[0] for item in models[0].named_hyperparameters()]
    params = []
    for model in models:
        params += list(model.parameters())
        
    opt = torch.optim.Adam(params=params,lr=.1)
    
    trainLosses = torch.zeros((nEpochs,1))
    testLosses = torch.zeros((nEpochs,1))
    trueLosses = torch.zeros((nEpochs,1))
    paramsList = []
    for epoch in range(nEpochs):
    
        opt.zero_grad()    
    
        yHat = splitPred(xtr,xList,models)
        loss = torch.mean((yHat-ytr)**2)
        trainLosses[epoch] = loss.detach()
        
        testLoss = torch.mean((splitPred(xte,xList,models)-yte)**2)
        testLosses[epoch] = testLoss.detach()
        
        trueLoss = torch.mean((splitPred(xtrue,xList,models)-ytrue)**2)
        trueLosses[epoch] = trueLoss.detach()
        
        loss.backward()
        
        opt.step()
        
        params = []
        for model in models:
            params += list(model.parameters())
        paramsList.append(params)
    
    # choose the params with loses true loss
    minInd = torch.argmin(trueLosses)
    bestParams = paramsList[minInd]
    
    bestModels = []
    for i,model in enumerate(models):
        _model = model
        _model.initialize(**dict(zip(paramNames,bestParams[i*len(paramNames):(i+1)*len(paramNames)])))
        bestModels.append(_model)
    
    return tuple(bestModels),trainLosses,testLosses,trueLosses

# split and try training new models
splitPoint = torch.mean(xtrain)
x_L,x_R,labels = split(xtrain,splitPoint)
y_L,y_R = ytrain[labels],ytrain[~labels]

xtrue_L,xtrue_R = x_L[::2],x_R[::2]
ytrue_L,ytrue_R = y_L[::2],y_R[::2]

model_L,_,_ = trainGP(x_L, y_L, xtrue_L, ytrue_L, nEpochs=1000)
model_R,_,_ = trainGP(x_R, y_R, xtrue_R, ytrue_R, nEpochs=1000)

# fit the params of the splitting model by minimizing MSE on the training data
# models,trainLosses,testLosses,trueLosses = fitParams(xtrain, ytrain,
#                                                       xtest, ytest,
#                                                       xtrue, ytrue,
#                                                       xList=(x_L,x_R),
#                                                       models=(model,model_L,model_R),
#                                                       nEpochs=1000)

# models,losses = trainSplitGP(xList=(x_L,x_R),
#                      yList=(y_L,y_R),
#                      xModels=(x_L,x_R),
#                      models=(model,model_L,model_R),
#                      nEpochs=1000)

# fig,ax = plt.subplots(1)
# ax.plot(losses)
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Negative Log Likelihood')
# ax.set_title('Split GP Negative Log Likelihood Losses')

# fig,ax = plt.subplots(1)
# ax.plot(trainLosses,label='Training Loss',alpha=.5)
# ax.plot(trainLosses,label='Test Loss',alpha=.5)
# ax.plot(trueLosses,label='True Loss (w/ noiseless response)',alpha=.5)
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss (MSE)')
# ax.set_title('Split GP Losses')
# ax.legend()

# model,model_L,model_R = models

epanKernel = EpanechnikovKernel(ard_num_dims=1)
epanKernel.lengthscale = torch.tensor([1.0])
pmodel = ExactGPModel(x,y,gpytorch.likelihoods.GaussianLikelihood())
pmodel.covar_module = epanKernel
yHat = splitPred(xtrue,xList=(x_L,x_R),models=(pmodel,model_L,model_R))

splitMse = torch.mean((yHat-ytrue)**2)

fig,axes = plt.subplots(1,4)
cmap = axes[0].plot(xtrue.detach(),ytrue)
axes[1].plot(x.detach(),y.detach())

# for i in range(nSteps**2):
#     axes[0].arrow(gradGrid[i,0].numpy(),gradGrid[i,1].numpy(),trueGradsSc[i,0].numpy(),trueGradsSc[i,1].numpy(),color='orange')
axes[2].plot(xtrue,f_pred.mean.detach())
axes[2].scatter(xtrain,ytrain,color='orange')
axes[3].plot(xtrue,yHat.detach())
axes[3].scatter(xtrain,ytrain,color='orange')
# for i in range(nSteps**2):
#     axes[1].arrow(gradGrid[i,0].numpy(),gradGrid[i,1].numpy(),estGradsSc[i,0].numpy(),estGradsSc[i,1].numpy(),color='orange')
    
axes[0].set_title('True function')
axes[1].set_title('Noisy function')
axes[2].set_title('Estimated function w/ Regular GP')
axes[3].set_title('Best (True Loss) Estimated function w/ Split GP')

fig,axes = plt.subplots(1,4)
cmap = axes[0].plot(xtrue.detach(),ytrue)
axes[1].plot(x.detach(),y.detach())
axes[1].scatter(xtrain,ytrain,color='orange',alpha=.5)

axes[2].plot(xtrue,model_L(xtrue).mean.detach())
axes[2].scatter(x_L,y_L,color='orange',alpha=.5)

axes[3].plot(xtrue,model_R(xtrue).mean.detach())
axes[3].scatter(x_R,y_R,color='orange',alpha=.5)

axes[0].set_title('True function')
axes[1].set_title('Noisy function')
axes[2].set_title('Left Child Model')
axes[3].set_title('RightChild Model')

# # plot the predictions
# fig,axes = plt.subplots(1,4)
# cmap = axes[0].contourf(x1.detach(),x2.detach(),yTrue.reshape((100,100)))
# axes[1].contourf(x1.detach(),x2.detach(),y.reshape((100,100)))
# axes[2].contourf(x1.detach(),x2.detach(),f_pred.mean.reshape((100,100)).detach())
# axes[3].contourf(x1.detach(),x2.detach(),yHat.reshape((100,100)).detach())
    
# axes[0].set_title('True function')
# axes[1].set_title('Noisy Function')
# axes[2].set_title('Estimated function w/ single GP\n MSE: {}'.format(mse))
# axes[3].set_title('Estimated function w/ split GP\n MSE: {}'.format(splitMse))
# axes[3].arrow(center[0],center[1],splitVec[-1][0],splitVec[-1][1],color='white',edgecolor='black')
# axes[3].scatter(X_train_L[:,0],X_train_L[:,1],color='orange',s=1)
# axes[3].scatter(X_train_R[:,0],X_train_R[:,1],color='blue',s=1)
# plt.colorbar(cmap)