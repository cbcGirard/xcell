#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:43:57 2022

@author: benoit
"""

import numba as nb
nb.config.DISABLE_JIT=0

import xCell
import numpy as np



nX=xCell.util.MAXPT

origin=np.zeros(3)
span=np.ones(3)
max_depth=10
bbox=np.concatenate((origin,span))

max_depth=xCell.util.max_depth

def testSyntheticElement():
    # for depth in range(1,20):
    for depth in range(max_depth,1,-1):
        for octN in range(8):
    
            elist=depth*[octN]
            eArr=np.array(elist)
            
            elOrigin=xCell.util.octant_list_to_xyz(eArr)
            pos=np.array([[c,b,a] for a in range(3) for b in range(3) for c in range(3)],
                         dtype=np.uint64)
            xyz=xCell.util.calc_xyz_within_octant(eArr, pos)
            stepInds=xCell.util.position_to_index(xyz, nX)
            inds=xCell.util.get_indices_of_octant(eArr,pos)
            assert np.equal(inds,stepInds).all()
            
            nupos=[xCell.util.index_to_xyz(ii, nX) for ii in inds]
            scale=2**(max_depth-depth)
            
            assert np.equal(nupos,pos*scale+elOrigin).all()
        

def testSyntheticMesh(max_depth):
    
    metric=xCell.makeExplicitLinearMetric(max_depth, 0.2)
    
    setup=xCell.Simulation("", bbox)
    setup.make_adaptive_grid(metric, max_depth)
    
    # setup.mesh.element_type='Face'
    
    setup.finalize_mesh()
    
    for el in setup.mesh.elements:
        indV=el.vertices
        coordV=xCell.util.indices_to_coordinates(indV, origin, span)
        assert np.equal(coordV[0],el.origin).all()
        assert np.equal(coordV[-1],el.origin+el.span).all()
        
        indF=el.faces
        coordF=xCell.util.indices_to_coordinates(indF, origin, span)
        
        
        assert np.equal(coordF[-1],el.center).all(), str(coordF[-1])+'=/='+str(el.center)
        
    return setup
    
max_depth=5
# testSyntheticElement()
setup=testSyntheticMesh(max_depth)