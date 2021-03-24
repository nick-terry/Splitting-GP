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
import VarSplittingGP as vsg

gpytorch.settings.debug._state = False
                
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=1)
        
        self.train_x = train_x
        self.train_y = train_y
        
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

x,y,xtrain,ytrain,xtest,ytest,xtrue,ytrue = getData2()    

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

f_pred = model(xtrue)
mse = torch.mean((f_pred.mean-ytrue)**2)

# split and try training new models
splitPoint = torch.mean(xtrain)
x_L,x_R,labels = split(xtrain,splitPoint)
y_L,y_R = ytrain[labels],ytrain[~labels]

xtrue_L,xtrue_R = x_L[::2],x_R[::2]
ytrue_L,ytrue_R = y_L[::2],y_R[::2]

# Create child models
model_L = ExactGPModel(x_L,y_L,gpytorch.likelihoods.GaussianLikelihood())
model_R = ExactGPModel(x_R,y_R,gpytorch.likelihoods.GaussianLikelihood())
children = [model_L,model_R]

varSplitModel = vsg.VariationalSplittingGPModel(xtrain,ytrain,children)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

opt = torch.optim.Adam(params=varSplitModel.parameters(),lr=.1)
mll = gpytorch.mlls.VariationalELBO(likelihood, varSplitModel, ytrain.numel())

for epoch in range(1000):
    
    opt.zero_grad()
    
    output = varSplitModel(xtrain)
    loss = -mll(output,ytrain)
    loss.backward()
    
    print('Loss: {}'.format(loss))
    
    opt.step()

varSplitModel.eval()
yHat = varSplitModel(xtrue).mean
splitMse = torch.mean((yHat-ytrue)**2)

fig,axes = plt.subplots(1,4)
cmap = axes[0].plot(xtrue.detach(),ytrue)
axes[1].plot(x.detach(),y.detach())

axes[2].plot(xtrue,f_pred.mean.detach())
axes[2].scatter(xtrain,ytrain,color='orange')
axes[3].plot(xtrue,yHat.detach())
axes[3].scatter(xtrain,ytrain,color='orange')
axes[0].set_title('True function')
axes[1].set_title('Noisy function')
axes[2].set_title('Estimated function w/ Regular GP\n MSE:{:.3f}'.format(mse))
axes[3].set_title('Best (True Loss) Estimated function w/ Split GP\n MSE:{:.3f}'.format(splitMse))

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

# Create splitting model w/ n evenly spaced children
def getSplitModel(x,y,n):
      
    # split data
    splitIndices = []
    xList,yList = [],[]
    size = x.size()[0]
    _splitInd = 0
    
    for i in range(0,n):
        splitInd = i * (size//n)
        splitIndices.append(splitInd)

        if i == 0:
            xi,yi = x[:splitInd],y[:splitInd]
        elif i < n-1:
            xi,yi = x[_splitInd:splitInd],y[_splitInd:splitInd]
        else:
            xi,yi = x[_splitInd:],y[_splitInd:]
        
        xList.append(xi)
        yList.append(yi)
    
        _splitInd = splitInd    
    
    # train models
    children = []
    for i in range(0,n):
        
        model = ExactGPModel(xList[i],yList[i],gpytorch.likelihoods.GaussianLikelihood())
        children.append(model)
    
    varSplitModel = vsg.VariationalSplittingGPModel(x,y,children)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    opt = torch.optim.Adam(params=varSplitModel.parameters(),lr=.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, varSplitModel, y.numel())
    
    for epoch in range(1000):
    
        opt.zero_grad()
        
        output = varSplitModel(x)
        loss = -mll(output,y)
        loss.backward()
        
        print('Loss: {}'.format(loss))
        
        opt.step()
        
    varSplitModel.eval()
    
    return varSplitModel,children,xList,yList

n=5
splitModelN,children,xList,yList = getSplitModel(xtrain, ytrain, n)
yHatN = splitModelN(xtrue).mean
splitMseN = torch.mean((yHatN-ytrue)**2)

fig,axes = plt.subplots(1,4)
cmap = axes[0].plot(xtrue.detach(),ytrue)
axes[1].plot(x.detach(),y.detach())

axes[2].plot(xtrue,f_pred.mean.detach())
axes[2].scatter(xtrain,ytrain,color='orange')
axes[3].plot(xtrue,yHatN.detach())
axes[3].scatter(xtrain,ytrain,color='orange')
axes[0].set_title('True function')
axes[1].set_title('Noisy function')
axes[2].set_title('Estimated function w/ Regular GP\n MSE:{:.3f}'.format(mse))
axes[3].set_title('Best (True Loss) Estimated function w/ Split GP\n MSE:{:.3f}'.format(splitMseN))

# fig,axes = plt.subplots(1,n)
# cmap = axes[0].plot(xtrue.detach(),ytrue)
# axes[1].plot(x.detach(),y.detach())
# axes[1].scatter(xtrain,ytrain,color='orange',alpha=.5)

# for i in range(n):
#     axes[i].plot(xtrue,children[i](xtrue).mean.detach())
#     axes[i].scatter(xList[i],yList[i],color='orange',alpha=.5)

#     axes[i].set_title('{}th Child Model'.format(i))

# axes[0].set_title('True function')
# axes[1].set_title('Noisy function')

