#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:43:57 2022

@author: benoit
"""

import xCell
import numpy as np

nX=xCell.util.MAXPT

origin=np.zeros(3)
span=np.ones(3)
maxdepth=6
bbox=np.concatenate((origin,span))

MAXDEPTH=xCell.util.MAXDEPTH

def testSyntheticElement():
    # for depth in range(1,20):
    for depth in range(MAXDEPTH,1,-1):
        for octN in range(8):
    
            elist=depth*[octN]
            eArr=np.array(elist)
            
            elOrigin=xCell.util.octantListToXYZ(eArr)
            pos=np.array([[c,b,a] for a in range(3) for b in range(3) for c in range(3)],
                         dtype=np.uint64)
            xyz=xCell.util.xyzWithinOctant(eArr, pos)
            stepInds=xCell.util.pos2index(xyz, nX)
            inds=xCell.util.indicesWithinOctant(eArr,pos)
            assert np.equal(inds,stepInds).all()
            
            nupos=[xCell.util.index2pos(ii, nX) for ii in inds]
            scale=2**(MAXDEPTH-depth)
            
            assert np.equal(nupos,pos*scale+elOrigin).all()
        


































def metric(coords):
    r=np.linalg.norm(coords)/16
    return r

setup=xCell.Simulation("", bbox)
setup.makeAdaptiveGrid(metric, maxdepth)

# setup.mesh.elementType='Face'

setup.finalizeMesh()

for el in setup.mesh.elements:
    indV=el.vertices
    coordV=xCell.util.indexToCoords(indV, origin, span)
    assert np.equal(coordV[0],el.origin).all()
    assert np.equal(coordV[-1],el.origin+el.span).all()
    
    indF=el.faces
    coordF=xCell.util.indexToCoords(indF, origin, span)
    
    
    assert np.equal(coordF[-1],el.center).all(), str(coordF[-1])+'=/='+str(el.center)