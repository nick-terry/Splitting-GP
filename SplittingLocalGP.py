# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:52:23 2020

@author: pnter
"""
from LocalGP import LocalGPModel,LocalGPChild
from UtilityFunctions import pddp
import numpy as np
import multiprocessing as mp
import torch

'''
Workaround for a invalid incompatability warning between numpy 1.8.1 and scipy 0.18.1
'''
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

'''
A Local GP model which recursively splits child models as the number of
data points grows. This preserves computational tractability as the model grows,
while reducing prediction MSE as compared to naive generation of new child models.
'''
class SplittingLocalGPModel(LocalGPModel):
    def __init__(self, likelihoodFn, kernel, splittingLimit, inheritKernel=True, **kwargs):
        super(SplittingLocalGPModel,self).__init__(likelihoodFn, kernel, **kwargs)
        
        self.splittingLimit = splittingLimit

    '''
    TODO: override this method to allow for multiple new child models in the
    event of a split
    '''
    def update(self, x, y):
        #If no child model have been created yet, instantiate a new child with {x,y} and record the output dimension
        if len(self.children)==0:
            self.createChild(x,y)
            self.outputDim = int(y.shape[-1])
            
        #If child models exist, find the the child whose center is closest to x
        else:
            '''
            TODO: for the SplittingLocalGPModel, we may be able to optimize this search
            by using a special data structure such as a binary tree
            '''
            closestChildIndex,minDist = self.getClosestChild(x)
                
            closestChildModel = self.children[closestChildIndex]
            
            #Create new model(s) which additionally incorporates the pair {x,y}. This will return more than one model
            #if a split occurs.
            newChildModels = closestChildModel.update(x,y)
            
            #Replace the existing model with the new model(s) which incorporates new data
            del self.children[closestChildIndex]
            self.addChildren(newChildModels)
            del closestChildModel
    
    '''
    Add one or more child models
    '''
    def addChildren(self,children):
        if type(children) is not list:
            children = [children] 
        self.children += children
        
        #Due to the order of training, the last child updated is the last child in the list
        self.setChildLastUpdated(self.children[-1])
        
        #Need to update the parent
        for child in self.children:
            child.parent = self
    
    '''
    Instantiate a new child model using the training pair {x,y}
    
    Note that the likelihood used to instantiate the child model is distinct
    from each other child model, as opposed to the kernel which is shared 
    between the children.
    '''
    def createChild(self,x,y):
        #Create new child model
        newChildModel = SplittingLocalGPChild(x,y,self,self.inheritKernel)

        #Add to the list of child models
        self.children.append(newChildModel)
        
class SplittingLocalGPChild(LocalGPChild):
    def __init__(self, train_x, train_y, parent, inheritKernel=True):
        super(SplittingLocalGPChild,self).__init__(train_x, train_y, parent, inheritKernel)
        
        
    '''
    Split the child model into k smaller models. Divide the inputs amongst the 
    models using Lloyd's algorithm to compute an approximate k-means clustering.
    
    Returns a list containing the new child models.
    '''
    def split(self,k=2):
        #Detach inputs,targets for k-means
        train_xDet = self.train_x.detach()
        train_yDet = self.train_y.detach()
        
        '''
        Split the inputs into clusters using PDDP
        '''
        labels = pddp(train_xDet)
        
        #Organize the training inputs, targets, centroids for splitting
        train_x_list = [train_xDet[labels==i] for i in range(k)]
        train_y_list = [train_yDet[labels==i] for i in range(k)]
        
        newChildrenArgs=[(train_x,train_y,self.parent,self.parent.inheritKernel) for train_x,train_y in zip(train_x_list,train_y_list)]
        newChildren = []
        
        for args in newChildrenArgs:
            newChildren.append(SplittingLocalGPChild(*args))
            
            for child in newChildren:
                print(child.covar_module.state_dict())
        
        return newChildren
    
    '''
    Update the child model with new pair {x,y}. If the model has become larger
    than the splitting threshold, split it. By default, it will be split in half.
    '''
    def update(self, x, y):
        #Check if the new pair will make the model larger than the threshold
        if self.train_x.shape[0] >= self.parent.splittingLimit:
            print('Splitting a local child model...')
            return self.split()
        
        else:
            '''
            If this was the last child to be updated, or inheritKernel=False, we can do a fantasy update
            without updating the covar cache. Otherwise, we can do a rank-one update of the covar cache
            prior to the fantasy update.
            '''
            if not (self.lastUpdated or self.parent.inheritKernel==False):
                self.updateInvCovarCache()
            
            '''
            Sometimes get an error when attempting Cholesky decomposition.
            In this case, refit a new model.
            '''
            try:
                updatedModel = self.get_fantasy_model(inputs=x, targets=y)
            
            except RuntimeError as e:
                print('Error during Cholesky decomp for fantasy update. Fitting new model...')
                raise e
                newInputs = torch.cat([self.train_x,x],dim=0)
                newTargets = torch.cat([self.train_y,y],dim=0)
                updatedModel = SplittingLocalGPChild(newInputs,newTargets,self.parent)
            
            except RuntimeWarning as e:
                print('Error during Cholesky decomp for fantasy update. Fitting new model...')
                raise e
                newInputs = torch.cat([self.train_x,x],dim=0)
                newTargets = torch.cat([self.train_y,y],dim=0)
                updatedModel = SplittingLocalGPChild(newInputs,newTargets,self.parent)
                
                
            #Update the data properties
            updatedModel.train_x = updatedModel.train_inputs[0]
            updatedModel.train_y = updatedModel.train_targets
            
            #Compute the center of the new model
            updatedModel.center = torch.mean(updatedModel.train_x,dim=0)
            
            #Need to perform a prediction so that get_fantasy_model may be used to update later
            updatedModel.predict(x)
            return updatedModel