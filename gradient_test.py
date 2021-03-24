# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:05:26 2020

@author: pnter
es"""
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def getData():
    
    x = torch.linspace(-2,2,steps=100)
    x1,x2 = torch.meshgrid([x,x])
    # y = x1**2
    y = x1**2 + 2*x2
    # y = (x1-2)**2 + (x2-2)**2 + 100*torch.max(x1,torch.zeros((100,100))) + 100*torch.max(x2,torch.zeros((100,100)))
    rate = 1
    dist1 = torch.distributions.exponential.Exponential(rate, validate_args=None)
    dist2 = torch.distributions.exponential.Exponential(rate/2, validate_args=None)
    noise1 = dist1.sample(y.shape)
    noise2 = torch.zeros(y.shape)
    noise2[:50,:50] = dist2.sample((50,50))
    return x1,x2,y,y+noise1+noise2

def trueGrad(x):
    #return torch.tensor([2,0]) * x
    return torch.tensor([2,0]) * x + torch.tensor([0,2])

def estGrad(x,model):
    k = model.covar_module(x,X_train).evaluate()
    Kinvy = torch.solve(y_train[:,None],model.covar_module(X_train,X_train).evaluate()).solution
    logLs = model.covar_module.base_kernel.raw_lengthscale
    thetaEye = torch.diag(torch.exp(logLs).squeeze())
    
    if x.shape[0] > 1:
        delta  = X_train[None,:,:] - x[:,None,:]
        kd = (k[:,:,None] * delta)
    else:    
        delta = X_train - x
        kd = (k * delta)
    
    return Kinvy.T.matmul(kd).matmul(thetaEye).squeeze()

def getC(grads):
    return torch.sum(grads[:,None,:]*grads[:,:,None],dim=0)

def split(X,splitVec):
    labels = torch.sum(X*splitVec,dim=1)<0
    return X[labels],X[~labels],labels

#Fix the random seeds for using CRN with rBCM experiment
torch.manual_seed(10101)
np.random.seed(10101)

x1,x2,yTrue,y = getData()
X = torch.stack((x1,x2),dim=-1).reshape((100*100,2))
yTrue = torch.reshape(yTrue,(100*100,1)).flatten()
y = torch.reshape(y,(100*100,1)).flatten()

trainInd = torch.multinomial(torch.tensor(range(100*100)).float(),num_samples=500)

X_train,y_train = X[trainInd,:],y[trainInd]

def trainGP(X_train,y_train):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)
    
    model.train()
    likelihood.train()
    
    # set up mll and optimizer
    opt = torch.optim.Adam(model.parameters(), lr=.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    #Up this setting to prevent potential numerical issues if CG hasn't converged in <2000 iterations
    for epoch in range(50):
        
        opt.zero_grad()
        output = model(X_train)
        
        loss = -mll(output,y_train)
        loss.backward()
        
        print('Epoch {} has loss: {}'.format(epoch,loss))
        
        opt.step()
    
    model.eval()
    
    return model

model = trainGP(X_train, y_train)

f_pred = model(X)

# compute some gradients
nSteps = 10
x = torch.linspace(-1,1,steps=nSteps)
z1,z2 = torch.meshgrid([x,x])
gradGrid = torch.stack((z1,z2),dim=-1).reshape((nSteps**2,2))
trueGrads = trueGrad(gradGrid)
trueGradsSc = trueGrads/torch.norm(trueGrads,p=2,dim=1,keepdim=True)/25
estGrads = estGrad(gradGrid, model).detach()
estGradsSc = estGrads/torch.norm(estGrads,p=2,dim=1,keepdim=True)/25

# fig,axes = plt.subplots(1,2)
# cmap = axes[0].contourf(x1.detach(),x2.detach(),y.reshape((100,100)))
# for i in range(nSteps**2):
#     axes[0].arrow(gradGrid[i,0].numpy(),gradGrid[i,1].numpy(),trueGradsSc[i,0].numpy(),trueGradsSc[i,1].numpy(),color='orange')
# axes[1].contourf(x1.detach(),x2.detach(),f_pred.mean.reshape((100,100)).detach())
# for i in range(nSteps**2):
#     axes[1].arrow(gradGrid[i,0].numpy(),gradGrid[i,1].numpy(),estGradsSc[i,0].numpy(),estGradsSc[i,1].numpy(),color='orange')
    
# axes[0].set_title('True function/gradients')
# axes[1].set_title('Estimated function/gradients w/ GP')
# plt.colorbar(cmap)


# compute MSE of grad vectors
mse = torch.mean((f_pred.mean-yTrue)**2)
gradMse = torch.mean(torch.norm(trueGrads-estGrads,p=2,dim=1),dim=0)

# try using eigenvector of C matrix to split
C = getC(estGrads)
# eigenvalues are sorted in ascending order, so take the last eigenvector to split
Lambda,eigV = torch.symeig(C,eigenvectors=True)
center = torch.mean(X_train,dim=0)



# split and try training new models
splitVec = eigV[-1,None]
X_train_L,X_train_R,labels = split(X_train,splitVec)
y_train_L,y_train_R = y_train[labels],y_train[~labels]

model_L = trainGP(X_train_L, y_train_L)
model_R = trainGP(X_train_R, y_train_R)

# compute predictions with the new local models
yHat_L = model_L(X).mean
yHat_R = model_R(X).mean

X_L,X_R,_ = split(X,splitVec)

K_L = torch.sum(model.covar_module(X,X_L).evaluate(),dim=1)
K_R = torch.sum(model.covar_module(X,X_R).evaluate(),dim=1)

yHat = (yHat_L * K_L + yHat_R * K_R)/(torch.sum(K_L + K_R))
splitMse = torch.mean((yHat-y)**2)

# plot the predictions
fig,axes = plt.subplots(1,4)
cmap = axes[0].contourf(x1.detach(),x2.detach(),yTrue.reshape((100,100)))
axes[1].contourf(x1.detach(),x2.detach(),y.reshape((100,100)))
axes[2].contourf(x1.detach(),x2.detach(),f_pred.mean.reshape((100,100)).detach())
axes[3].contourf(x1.detach(),x2.detach(),yHat.reshape((100,100)).detach())
    
axes[0].set_title('True function')
axes[1].set_title('Noisy Function')
axes[2].set_title('Estimated function w/ single GP\n MSE: {}'.format(mse))
axes[3].set_title('Estimated function w/ split GP\n MSE: {}'.format(splitMse))
axes[3].arrow(center[0],center[1],splitVec[-1][0],splitVec[-1][1],color='white',edgecolor='black')
axes[3].scatter(X_train_L[:,0],X_train_L[:,1],color='orange',s=1)
axes[3].scatter(X_train_R[:,0],X_train_R[:,1],color='blue',s=1)
plt.colorbar(cmap)