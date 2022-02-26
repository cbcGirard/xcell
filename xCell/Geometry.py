#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:07:45 2022

@author: benoit
"""

import numpy as np
import numba as nb
from numba import int64, float64


@nb.experimental.jitclass([
    ('center',float64[:]),
    ('radius', float64)
    ])
class Sphere:
    def __init__(self,center,radius):
        self.center=center
        self.radius=radius
        
    def isInside(self,coords):
        N=coords.shape[0]
        isIn=np.empty(N,dtype=np.bool_)
        for ii in nb.prange(N):
            c=coords[ii]
            isIn[ii]=np.linalg.norm(c-self.center)<=self.radius
            
        return isIn
            
    
@nb.experimental.jitclass([
    ('center',float64[:]),
    ('radius', float64),
    ('length',float64),
    ('axis',float64[:])
    ])
class Cylinder:
    def __init__(self,center,radius,length,axis):
        self.center=center
        self.radius=radius
        
    def isInside(self,coords):
        N=coords.shape[0]
        isIn=np.empty(N,dtype=np.bool_)
        for ii in nb.prange(N):
            c=coords[ii]
            isIn[ii]=np.linalg.norm(c-self.center)<=self.radius
            
        return isIn