#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:59:01 2021

@author: benoit
"""

import xCell
import numpy as np
import matplotlib.pyplot as plt
import numba as nb

xmax=1e-4

maxDepth=3

sigma=np.ones(3,dtype=np.float64)
vMode=True

bbox=np.concatenate((-xmax*np.ones(3), xmax*np.ones(3)))

otree=xCell.Octree(bbox,maxDepth=maxDepth)
setup=xCell.Simulation()



def inverseSquare(coord):
    return 1/np.dot(coord,coord)

def toR(coord):
    return 0.5*np.linalg.norm(coord)

otree.refineByMetric(toR)
# otree.printStructure()
numEl=otree.countElements()
octs=otree.tree.getAllChildren()

rawCoords,rawIndices=otree.tree.getCoordsRecursively()
# plt.plot(indices,np.array(coords),linestyle='None',marker='.')

# _,globalMap=np.unique(rawIndices,return_index=True)
# coords=np.array([rawCoords[ii] for ii in globalMap])
coords=np.unique(rawCoords,axis=0,return_inverse=False)
indices=np.array([otree.coord2Index(c) for c in coords])

print(len(indices))
print(len(np.unique(indices)))

ax=xCell.new3dPlot(bbox)

X,Y,Z=np.hsplit(coords,3)
ax.scatter3D(X,Y,Z,c=indices)
# for o in octs:
#     ocoords=o.getOwnCoords()
#     oNdx=np.array([otree.coord2Index(c) for c in ocoords])
#     globMask=np.isin(indices,oNdx,assume_unique=True)
#     globndx=indices[globMask]
#     # globs=[np.]
#     setup.mesh.addElement(o.origin, o.span, sigma, o.globalNodes)

# sourceIndex=otree.coord2Index(np.zeros(3))

# if vMode:
#     srcMag=1

#     setup.addVoltageSource(srcMag,index=sourceIndex)
#     srcType='Voltage'
# else:
#     srcMag=1e-9
#     setup.addCurrentSource(srcMag,index=sourceIndex)
#     srcType='Current'

# # ground boundary nodes
# for ii in nb.prange(coords.shape[0]):
#     pt=np.abs(coords[ii])
#     if np.any(pt==xmax):
#         setup.addVoltageSource(0,index=ii)
    
    
# edges,conductances=setup.mesh.getConductances()


# ax=xCell.new3dPlot(bbox)
# xCell.showEdges(ax, coords, edges, conductances)


# v=setup.solve()

# coordArray=np.unique(coords,axis=1)
# X,Y,Z=np.hsplit(coordArray,3)

# ax=xCell.new3dPlot(bbox)
# xCell.showNodes(ax, coordArray, np.arange(coordArray.shape[0]))