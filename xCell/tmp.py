#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:49:59 2021

@author: benoit
"""
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import xCell
import matplotlib.tri as tri


meshtype='adaptive'
studyPath='Results/studyTst/animateAlgo/'#+meshtype

maxdepth=12
xmax=1e-4
sigma=np.ones(3)


vMode=False
showGraphs=False
generate=False
saveGraphs=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)

nb.config.DEBUG_TYPEINFER=0

      



def getquads(x,y,xInt,yInt,values):
    # x,y=np.hsplit(xy,2)
    # x=xy[:,0]
    # y=xy[:,1]
    _,kx,nx=np.unique(xInt,return_index=True,return_inverse=True)
    _,ky,ny=np.unique(yInt,return_index=True,return_inverse=True)
    
    quadVals=[]
    quadCoords=[]
    
    sel=np.empty((4,x.shape[0]),dtype=np.bool8)
    
    for yy in nb.prange(len(ky)-1):
        for xx in nb.prange(len(kx)-1):
            
            sel=__getSel(nx, ny, xx, yy)
            
            if np.all(np.any(sel,axis=1)):
                # x0,y0=xy[sel[0]][0]
                # x1,y1=xy[sel[3]][0]
                x0=x[sel[0]][0]
                y0=y[sel[0]][0]
                x1=x[sel[3]][0]
                y1=y[sel[3]][0]
                qcoords=np.array([x0,x1,y0,y1])
                qvals=np.array([values[sel[n,:]] for n in np.arange(4)])

                quadVals.append(qvals)
                quadCoords.append(qcoords)
            
    return quadVals, quadCoords

@nb.njit()
def __getSel(x,y,x0,y0,k=0):
    sel=np.empty((4,x.shape[0]),dtype=np.bool8)
    
    sel[0]=np.logical_and(x==x0,y==y0)
    sel[1]=np.logical_and(x>(x0+k),y==y0)
    sel[2]=np.logical_and(x==x0,y>(y0+k))
    
    if np.any(sel[1]) & np.any(sel[2]):
        x1=x[sel[1]][0]
        y1=y[sel[2]][0]
        
        sel[3]=np.logical_and(x==x1,y==y1)
        
        if not np.any(sel[3]):
            sel=__getSel(x, y, x0, y0,k=k+1)
        else: 
            sel[1]=np.logical_and(x==x1,y==y0)
            sel[2]=np.logical_and(x==x0,y==y1)
    
    
    return sel
    
        
        
def intify(simulation):
    nx=simulation.ptPerAxis
    bb=simulation.mesh.bbox
    
    span=bb[3:]-bb[:3]
    float0=simulation.mesh.nodeCoords-bb[:3]
    ints=(nx*float0)/span
    
    return ints.astype(np.int64)
        
        

# pv=np.random.rand(100)
# pcoord=np.random.rand(100,2)

setup=study.loadData('sim20')

sel=setup.mesh.nodeCoords[:,-1]==0
intcoord=intify(setup)
pcoord=intcoord[sel,:-1]
ccoord=setup.mesh.nodeCoords[sel,:-1]
r=np.linalg.norm(ccoord,axis=1)

inside=r<1e-6
ana=np.empty_like(r)
ana[inside]=1
ana[~inside]=1e-6/r[~inside]

pv=setup.nodeVoltages[sel]
err=pv-ana



x,y=np.hsplit(ccoord, 2)
xi,yi=np.hsplit(pcoord,2)
qv,qc = getquads(x,y,xi,yi,err)

cMap,cNorm= xCell.getCmap(err)


ax=plt.figure().add_subplot()

ax.set_xlim(min(ccoord[:,0]),max(ccoord[:,0]))
ax.set_ylim(min(ccoord[:,1]),max(ccoord[:,1]))
   
edgePts=xCell.getPlanarEdgePoints(setup.mesh.nodeCoords, setup.edges)


for v,c in zip(qv,qc):
    
    # carts=[ccoord[kx[c[0]],0],
    #        ccoord[kx[c[1]],0],
    #        ccoord[ky[c[2]],1],
    #        ccoord[ky[c[3]],1]]
    plt.imshow(v.reshape((2,2)),
               origin='lower',extent=c.squeeze(),
                cmap=cMap, norm=cNorm,interpolation='bilinear')
    plt.show()
    
xCell.showEdges2d(ax, edgePts)
