# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:28:15 2020

@author: pnter
"""


from UtilityFunctions import pddp
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import numpy as np

#Generate some random points in 2-D
torch.random.manual_seed(42069)
points = torch.randn(size=(30,2))

labels,direction = pddp(points,True)

group1 = points[labels==1]
group2 = points[labels==0]
g1center = torch.mean(group1,dim=0)
g2center = torch.mean(group2,dim=0)



fig = plt.figure(figsize=(10,5))
ax = fig.subplots(1,2)

plt.subplots_adjust(wspace=0, hspace=0)

ax[0].scatter(points[:,0],points[:,1],c='blue',alpha=.5)
centroid = torch.mean(points,dim=0)
#Change to x instead of circle
ax[0].scatter(centroid[0],centroid[1],c='blue',edgecolor='black',marker='x')

ax[0].quiver(centroid[0],centroid[1],direction[0],direction[1],scale=3,
             facecolor='orange',
             edgecolor='black')

#Solve for orthogonal vector
A = torch.stack([direction,torch.ones((2))]).float()
b = torch.tensor([0,1]).unsqueeze(0).T.float()
ortho = b.solve(A).solution
ortho = ortho.squeeze()/torch.norm(ortho,2)


left = ortho*-3
right = ortho*3

ax[0].quiver([centroid[0],centroid[0]],[centroid[1],centroid[1]],[left[0],right[0]],[left[1],right[1]],scale=1,color='green')

ax[1].scatter(group1[:,0],group1[:,1],c='blue',alpha=.5)
ax[1].scatter(group2[:,0],group2[:,1],c='red',alpha=.5)

ax[1].scatter(g1center[0],g1center[1],c='blue',edgecolor='black',marker='x',s=100)
ax[1].scatter(g2center[0],g2center[1],c='red',edgecolor='black',marker='x',s=100)

'''
ax[0].set_xticks([],[])
ax[0].set_yticks([],[])

ax[1].set_xticks([],[])
'''
ax[1].set_yticks([],[])


ax[0].set_xlabel('X1')
ax[1].set_xlabel('X1')
ax[0].set_ylabel('X2')

plt.savefig('pca.pdf',dpi=300,bbox_inches='tight')

'''
ax[0].axis('off')
ax[1].axis('off')
'''

gridDims = 100
scale = 1
x,y = torch.meshgrid([torch.linspace(-scale,scale,gridDims), torch.linspace(-scale,scale,gridDims)])
xyGrid = torch.stack([x,y],dim=2).double()

#Evaluate a function to approximate
'''
z = (5*torch.sin((xyGrid[:,:,0]/scale+.5)**2+(2*xyGrid[:,:,1]/scale+.5)**2)+
     5*torch.sin((xyGrid[:,:,0]/scale-.5)**2+(2*xyGrid[:,:,1]/scale-.5)**2)).reshape((gridDims,gridDims,1))
'''
z = (5*torch.sin((xyGrid[:,:,0]/scale)**2+((2*xyGrid[:,:,1])/scale)**2)+3*xyGrid[:,:,0]/scale).reshape((gridDims,gridDims,1))
z -= torch.mean(z)
z += torch.randn(z.shape) * torch.max(z) * .05

maxAbsVal = torch.max(torch.abs(z))
levels = np.linspace(-maxAbsVal,maxAbsVal,30)

#Plot true function
fig2,axes = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(6,5))
contours = axes.contourf(xyGrid[:,:,0].detach(),xyGrid[:,:,1].detach(),z.detach().squeeze(2),levels)

plt.xlabel('X1',fontsize='x-large')
plt.ylabel('X2',fontsize='x-large')


plt.colorbar(contours)
fig2.savefig('synthetic_response.pdf',dpi=300,bbox_inches='tight')