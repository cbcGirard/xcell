#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:43:57 2022

@author: benoit
"""

import xCell
import numpy as np


origin=np.zeros(3)
span=np.ones(3)
maxdepth=16
bbox=np.concatenate((origin,span))


def metric(coords):
    r=np.linalg.norm(coords)/16
    return r

setup=xCell.Simulation("", bbox)
setup.makeAdaptiveGrid(metric, maxdepth)

# setup.mesh.elementType='Face'

setup.finalizeMesh()

for el in setup.mesh.elements:
    indV=el.calcVertexTags()
    coordV=xCell.util.indexToCoords(indV, origin, span, maxdepth)
    assert np.equal(coordV[0],el.origin).all()
    assert np.equal(coordV[-1],el.origin+el.span).all()
    
    indF=el.calcFaceTags()
    coordF=xCell.util.indexToCoords(indF, origin, span, maxdepth)
    
    
    assert np.equal(coordF[-1],el.center).all(), str(coordF[-1])+'=/='+str(el.center)