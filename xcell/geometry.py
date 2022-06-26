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
    ('center', float64[:]),
    ('radius', float64)
])
class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def isInside(self, coords):
        N = coords.shape[0]
        isIn = np.empty(N, dtype=np.bool_)
        for ii in nb.prange(N):
            c = coords[ii]
            isIn[ii] = np.linalg.norm(c-self.center) <= self.radius

        return isIn

# @nb.experimental.jitclass([
#     ('center',float64[:]),
#     ('radius', float64),
#     ('axis',float64[:])
#     ])


class Disk:
    def __init__(self, center, radius, axis, tol=1e-2):
        self.center = center
        self.radius = radius
        self.axis = axis/np.linalg.norm(axis)
        self.tol = tol

    def isInside(self, coords):
        N = coords.shape[0]
        isIn = np.empty(N, dtype=np.bool_)

        delta = coords-self.center
        deviation = np.dot(delta, self.axis)

        for ii in nb.prange(N):
            # c=coords[ii]

            # vec=c-self.center
            dist = np.linalg.norm(delta[ii])

            isIn[ii] = abs(deviation[ii]) < (
                self.radius*self.tol) and dist <= self.radius

        return isIn


# @nb.experimental.jitclass([
#     ('center',float64[:]),
#     ('radius', float64),
#     ('length',float64),
#     ('axis',float64[:])
#     ])
class Cylinder:
    def __init__(self, center, radius, length, axis):
        self.center = center
        self.radius = radius
        self.length = length
        self.axis = axis/np.linalg.norm(axis)

    def isInside(self, coords):
        N = coords.shape[0]
        isIn = np.empty(N, dtype=np.bool_)

        delta = coords-self.center
        deviation = np.dot(delta, self.axis)

        for ii in nb.prange(N):

            vec = delta[ii]
            dz = deviation[ii]
            dr = np.sqrt(np.dot(vec, vec)-dz**2)
            # dist=np.linalg.norm(vec)

            isIn[ii] = 0 <= dz <= self.length and dr <= self.radius
        return isIn
