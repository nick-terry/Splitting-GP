# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:57:53 2020

@author: pnter
"""

import torch
import gpytorch
# from gpytorch.utils.memoize import add_to_cache, is_in_cache
from gpytorch.lazy.root_lazy_tensor import RootLazyTensor
import copy
from UtilityFunctions import updateInverseCovarWoodbury
from math import inf

'''
Implements the Local Gaussian Process Regression Model as described by Nguyen-tuong et al.

Note that the kernel used in the original paper Local Gaussian Process Regression for Real Time Online Model Learning uses the RBF kernel

Parameters:
    likelihoodFn: The function which, when called, instantiates a new likelihood of the type which should be used for all child models
    kernel: The kernel function used to construct the covariances matrices
    w_gen: The threshold distance for generation of a new child model

'''
class LocalGPModel:
    def __init__(self, likelihoodFn, kernel, inheritKernel=True, **kwargs):        
        #Initialize a list to contain local child models
        self.children = []
        self.w_gen = kwargs['w_gen'] if 'w_gen' in kwargs else .5
        self.covar_module = kernel
        self.mean_module = kwargs['mean'] if 'mean' in kwargs else gpytorch.means.ConstantMean
        self.likelihood = likelihoodFn
        self.inheritKernel = inheritKernel
        
        #Number of training iterations used each time child model is updated
        #This should be roughly proportional to the number of observations.
        #By default, we will use 30. As number of data goes up, this may increase
        self.training_iter = 30
        
        #Default output dimension is 1 (scalar)
        self.outputDim = 1 if 'outputDim' not in kwargs else kwargs['outputDim']
        
        #If numInducingInputs is given, use variational GP models for child models
        if 'numInducingPoints' in kwargs:
            self.numInducingPoints = kwargs['numInducingPoints']
            assert(type(self.numInducingPoints)==int)
            assert(self.numInducingPoints>0)
            self.objectiveFunctionClass = gpytorch.mlls.VariationalELBO
        else:
            self.numInducingPoints = None
        
        #If maxChildren in kwargs, set self.maxChildren. Else, set to inf
        if 'maxChildren' in kwargs:
            self.maxChildren = kwargs['maxChildren']
        else:
            self.maxChildren = inf
        
        #If M=# of closest models for prediction is given, set parameter
        if 'M' in kwargs:
            self.M = kwargs['M']
        else:
            self.M = None
            
    '''
    Update the LocalGPModel with a pair {x,y}.
    '''   
    def update(self, x, y):
        #If no child model have been created yet, instantiate a new child with {x,y} and record the output dimension
        if len(self.children)==0:
            self.createChild(x,y)
            self.outputDim = int(y.shape[-1])
            
        #If child models exist, find the the child whose center is closest to x
        else:
            closestChildIndex,minDist = self.getClosestChild(x)
            
            #Get the mask of any points for which the closest model is not similar enough
            genNewModelIndices = (minDist < self.w_gen) if minDist.dim()>0 else (minDist < self.w_gen).unsqueeze(0)
            
            x_gen = x[genNewModelIndices,:]
            y_gen = y[genNewModelIndices]
            
            #Now generate a new model, if needed. 
            if x_gen.shape[0] > 0:
                
                self.createChild(x_gen[0,:].unsqueeze(0), y_gen[0].unsqueeze(0))
                
                #We then recursively call update() without the point which generated 
                #the model and return, in case some points would be assigned the newly generated model
                if x.shape[0] > 1:
                    x_minus = torch.cat([x[0:genNewModelIndices[0]], x[genNewModelIndices[0]:]])
                    y_minus = torch.cat([y[0:genNewModelIndices[0]], y[genNewModelIndices[0]:]])
                
                    self.update(x_minus,y_minus)
                
                return
                
            #Get points where we are not generating a new model
            x_assign = x[genNewModelIndices.bitwise_not()]
            y_assign = y[genNewModelIndices.bitwise_not()]
            
            closestIndex_assign = closestChildIndex[genNewModelIndices.bitwise_not()]\
                if closestChildIndex.dim()>0 else closestChildIndex.unsqueeze(0)[genNewModelIndices.bitwise_not()]
            
            #loop over children and assign them the new data points
            for childIndex in range(len(self.children)):
                #Get the data which are closest to the current child
                x_child = x_assign[closestIndex_assign==childIndex].squeeze(0)
                y_child = y_assign[closestIndex_assign==childIndex].squeeze(0)
                
                #If new data is a singleton, unsqueeze the 0th dim
                if x_child.dim() == 1:
                    x_child,y_child = x_child.unsqueeze(0),y_child.unsqueeze(0)
                     
                #Only proceed if there are some data in the batch assigned to the child
                if x_child.shape[0] > 0:
                    
                    closestChildModel = self.children[childIndex]
                    
                    #Create new model(s) which additionally incorporates the pair {x,y}. This will return more than one model
                    #if a split occurs.
                    newChildModel = closestChildModel.update(x_child,y_child)
                
                    #Replace the existing model with the new model which incorporates new data
                    self.children[closestIndex_assign] = newChildModel
            
            
            
        
    '''
    Instantiate a new child model using the training pair {x,y}
    
    Note that the likelihood used to instantiate the child model is distinct
    from each other child model, as opposed to the kernel which is shared 
    between the children.
    '''
    def createChild(self,x,y):
        #Create new child model, then train
        if self.numInducingPoints is None:
            newChildModel = LocalGPChild(x,y,self,self.inheritKernel)
        else:
            newChildModel = ApproximateGPChild(x,y,self,self.inheritKernel)
        
        #Set other children to not be last updated.
        self.setChildLastUpdated(newChildModel)
        
        #Add to the list of child models
        self.children.append(newChildModel)
    
    def setChildLastUpdated(self,child):
        for _child in self.children:
            _child.lastUpdated = False
        child.lastUpdated = True
            
    
    '''
    Return a pytorch tensor of the centers of all child models.
    '''
    def getCenters(self):
        #Get the center of each child model
        centersList = list(map(lambda x:x.center.reshape((x.center.shape[0])),self.children))
        #Return the centers after stacking in new dimension
        return torch.stack(centersList,dim=0)
    
    '''
    Returns the index of the closest child model to the point x, as well as the distance
    between the model's center and x.
    '''
    def getClosestChild(self,x):
        #Compute distances between new input x and existing inputs
        distances = self.getDistanceToCenters(x)
        #Get the single minimum distance from the tensor (max covar)
        minResults = torch.max(distances,1) if distances.dim()>1 else torch.max(distances,0)
        return minResults[1],minResults[0]
    
    '''
    Compute the distances from the point x to each center
    '''
    def getDistanceToCenters(self,x,returnPowers=False):
        centers = self.getCenters()
        x,centers = x.double(),centers.double()
        
        distances = self.covar_module(x,centers).evaluate()
        powers = torch.zeros(distances.shape)
        
        #Switch to double precision for this calculation
        '''
        vec = ((x-centers.repeat(x.shape[0],1))/self.covar_module.lengthscale).double().repeat(x.shape[0],1)
        powers = .5*torch.sum(vec**2,dim=1)
        distances = torch.exp(-powers)
        '''
        if returnPowers:
            return distances.squeeze(0),powers
        else:
            return distances.squeeze(0)
    
    '''
    Make a prediction at the point(s) x. This method is a wrapper which handles the messy case of multidimensional inputs.
    The actual prediction is done in the predictAtPoint helper method. If no M is given, use default
    '''
    def predict(self,x,individualPredictions=False,getVar=False):
        return self.predict_Helper(x,self.M,individualPredictions,getVar)
    
    '''
    Make a prediction at the point(s) x. This method is a wrapper which handles the messy case of multidimensional inputs.
    The actual prediction is done in the predictAtPoint helper method
    '''
    def predict_Helper(self,x,M,individualPredictions,getVar):
        if M is None:
            M = len(self.children)
        else:
            M = min(M,len(self.children))
        
        #Update all of the covar modules to the most recent
        if self.inheritKernel:  
            for child in self.children:
                child.covar_module = self.covar_module
        #If not inheriting kernel, then average the lengthscale params of child kernels
        else:
            lengthscales = [child.covar_module.lengthscale for child in self.children]
            self.covar_module.lengthscale = torch.mean(torch.stack(lengthscales),dim=0)
            
        mean_predictions = []
        var_predictions = []
        
        #Get the predictions of each child at each point
        for child in self.children:
            prediction = child.predict(x)
            
            mean_predictions.append(prediction.mean)
            var_predictions.append(prediction.variance)
        
        #Concatenate into pytorch tensors
        mean_predictions = torch.stack(mean_predictions).transpose(0,1)
        var_predictions = torch.stack(var_predictions).transpose(0,1)
        
        #Squeeze out any extra dims that may have accumulated
        if mean_predictions.dim()>2:
            mean_predictions = mean_predictions.squeeze()
            var_predictions = var_predictions.squeeze()
        
        #if the predictions are done at a single point, we need to unsqueeze in dim 0
        if mean_predictions.dim()<2:
            mean_predictions = mean_predictions.unsqueeze(-1)
            var_predictions = var_predictions.unsqueeze(-1)
        
        #Transpose to agree with minIndices dims
        #Note: This only needs to be done for the incremental experiments where we track memory usage.
        #Leave this commented out otherwise
        '''
        mean_predictions = mean_predictions.transpose(0,1)
        var_predictions = var_predictions.transpose(0,1)
        '''
        
        #We don't need this weighting procedure if there is only one child
        if mean_predictions.shape[-1]>1:
            #Get the covar matrix
            distances = self.getDistanceToCenters(x)
            
            #Get the M closest child models. Need to squeeze out extra dims of 1.
            sortResults = torch.sort(distances.squeeze(-1).squeeze(-1),descending=True)
            
            #Get the minDists for weighting predictions
            #minDists = sortResults[0][:,:M].squeeze(-1) if sortResults[0].dim()>1 else sortResults[0].unsqueeze(0)
            minDists = sortResults[0][:,:M] if sortResults[0].dim()>1 else sortResults[0].unsqueeze(0)
            
            #Get the min indices for selecting the correct predictions. If dim==1, then there is only one child, so no need to take up to M predictions
            minIndices = sortResults[1][:,:M] if sortResults[1].dim()>1 else sortResults[1].unsqueeze(0)
            
            #Get the associate predictions
            gatherDim = 1 if mean_predictions.dim()>1 else 0
            
            mean_predictions = mean_predictions.gather(gatherDim,minIndices)
            var_predictions = var_predictions.gather(gatherDim,minIndices)
            
            #Compute weights for the predictions. Switch to double precision for this somewhat unstable computation
            minDists = minDists.double()
            
            #If we have M=1, we need to unsqueeze for the summation
            if minDists.dim() == 1:
                minDists = minDists.unsqueeze(-1)
            
            #Sum the m smallest distances for each prediction point to normalize
            denominator = torch.sum(minDists,dim=1).unsqueeze(-1).repeat((1,minDists.shape[1]))

            weights = minDists/denominator
            
            
            #Compute weighted predictions.
            #IMPORTANT: the weighted variance predictions are highly negatively biased since we do not account for the covariance between models
            weighted_mean_predictions = torch.sum(weights * mean_predictions,dim=1)
            weighted_var_predictions = torch.sum(weights**2 * var_predictions,dim=1)
        
        else:
            weighted_mean_predictions = mean_predictions
            weighted_var_predictions = var_predictions

        if getVar:
            return weighted_mean_predictions,weighted_var_predictions
        elif individualPredictions:
            return weighted_mean_predictions,mean_predictions,weights,minDists
        else:
            return weighted_mean_predictions
        
    '''
    Make a prediction at the point x by finding the M closest child models and
    computing a weighted average of their predictions. By default M is the number
    of child models. If M < number of child models, use all of them.
    
    THIS METHOD IS NOW DEPRECATED. DO NOT RELY ON THIS.
    '''
    def predictAtPoint(self,x,M=None,individualPredictions=False):
        if M is None:
            M = len(self.children)
        else:
            M = min(M,len(self.children))
        
        #Compute distances between new input x and existing inputs
        distances,powers = self.getDistanceToCenters(x,True)
        
        #Get the M closest child models. Need to squeeze out extra dims of 1.
        sortResults = torch.sort(distances.squeeze(-1).squeeze(-1),descending=True)
        minDists = sortResults[0][:M].squeeze(-1) if sortResults[0].dim()>0 else sortResults[0].unsqueeze(0)
        minIndices = sortResults[1][:M] if sortResults[1].dim()>0 else sortResults[1].unsqueeze(0)
        closestChildren = [self.children[i] for i in minIndices]
        
        '''
        Get a posterior distribution for each child model. Note each will be
        multivariate normal. Then compute weighted average of the means of the
        posterior distributions.
        '''
        posteriorMeans = []
        for child in closestChildren:
            posterior = child.predict(x)
            posteriorMeans.append(posterior.mean)
        
        '''
        TODO: It would be better to instead compute the weighted average of the
        posterior distributions so we have access to variance as well.
        '''
        posteriorMeans = torch.stack(posteriorMeans)
        
        #We need to be careful with this computation. If the covariances are very small, we may end up with a nan value here.
        nonZeroDists = minDists[minDists>0.0]
        #Address the case where we are predicting very far away from all models. Take unweighted mean of all predictions
        if nonZeroDists.shape[-1]==0:
            weights = 1.0/(powers+1.0)
            weights = weights/torch.sum(weights)
        else:
            minDists = minDists
            weights = minDists/torch.sum(minDists)
        weightedAverageMean = torch.dot(weights,posteriorMeans.squeeze(-1).double()).float()
        
        if individualPredictions:
            return weightedAverageMean,posteriorMeans,weights,minDists

        else:
            return weightedAverageMean
        
class LocalGPChild(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, parent, inheritKernel=True, **kwargs):
        
        #Track if the child was created by splitting
        self.isSplittingChild = True if 'split' in kwargs and kwargs['split'] else False
        
        #Handle prior likelihood
        if 'priorLik' in kwargs and kwargs['priorLik'] is not None:
            priorLik =  kwargs['priorLik']
        else:
            #If no prior is provided, use the default of the parent
            priorLik = parent.likelihood()
            
            #In this case, we reset the isSplittingChild flag to false in order for the new likelihood to be trained
            self.isSplittingChild = False
            
        super(LocalGPChild, self).__init__(train_x, train_y, priorLik)
        
        #Set to double mode
        self.double()
        self.likelihood.double()
        
        self.parent = parent
        
        if 'priorMean' in kwargs and kwargs['priorMean'] is not None:
            #If given, take a prior for the mean. Used for splitting models.
            self.mean_module = copy.deepcopy(kwargs['priorMean'])
        else:
            self.mean_module = parent.mean_module()
        
        '''
        If inheritKernel is set to True, then the same Kernel function (including the same hyperparameters)
        will be used in all of the child models. Otherwise, a separate instance of the same kernel function
        is used for each child model.
        '''
        if inheritKernel:
            self.covar_module = parent.covar_module
        else:
            self.covar_module = parent.covar_module.__class__(ard_num_dims=train_x.shape[1] if train_x.dim()>1 else 1)
        self.lastUpdated = True
        
        '''
        Compute the center as the mean of the training data
        '''
        self.center = torch.mean(train_x,dim=0)
        if self.center.dim()==0:
            self.center = self.center.unsqueeze(0)
        
        self.train_x = train_x
        self.train_y = train_y
        
        self.trained = False
        
        self.initTraining()
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def update(self,x,y):
        #Sync covar
        if self.parent.inheritKernel:
            self.covar_module = self.parent.covar_module
        
        #Update train_x, train_y
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, y])
        
        #Update the data which can be used for optimizing
        self.train_inputs = (self.train_x,)
        self.train_targets = self.train_y
        
        #Flag the child as not having been trained.
        self.trained = False
        
        #Update center
        self.center = torch.mean(self.train_x,dim=0)
        if self.center.dim()==0:
            self.center = self.center.unsqueeze(0)
        
        return self
    
    '''
    Perform a rank-one update of the child model's inverse covariance matrix cache.
    '''
    def updateInvCovarCache(self,update=False):
        lazy_covar = self.prediction_strategy.lik_train_train_covar
        if is_in_cache(lazy_covar,"root_inv_decomposition"):
            if update:
                #Get the old cached inverse covar matrix 
                K_0inv = lazy_covar.root_inv_decomposition()
                #Get the new covar matrix by calling the covar module on the training data
                K = self.covar_module(self.train_x)
                #Compute the update
                Kinv = updateInverseCovarWoodbury(K_0inv, K)
                #Store updated inverse covar matrix in cache
                add_to_cache(lazy_covar, "root_inv_decomposition", RootLazyTensor(torch.sqrt(Kinv)))
            else:
                #This is a bit dirty, but here we will simply delete the root/root_inv from cache. This forces
                #GPyTorch to recompute them.
                
                lazy_covar._memoize_cache = {}
                self.prediction_strategy._memoize_cache = {}
                
    '''
    Setup optimizer and perform initial training
    '''
    def initTraining(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        #We only train on instantiation if the child model is not a result of a split
        if not self.isSplittingChild:
            #Setup optimizer
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
            mll.double()
            
            #Perform training iterations
            training_iter = self.parent.training_iter
            for i in range(training_iter):
                self.optimizer.zero_grad()
                output = self(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                    
                self.optimizer.step()
            
        self.trained = True
    
    '''
    Retrain model after new data is obtained
    '''
    def retrain(self):
        #Switch to training mode
        self.train()
        self.likelihood.train()
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        #Perform training iterations
        training_iter = self.parent.training_iter
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()

            self.optimizer.step()
        
        self.trained = True
    
    '''
    Evaluate the child model to get the predictive posterior distribution
    '''
    def predict(self,x):
        
        if not self.trained:
            self.retrain()
        
        #Switch to eval/prediction mode
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self(x))
        
        return prediction