#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:39:59 2022
Element type routines
@author: benoit
"""
import numpy as np
import numba as nb
from numba import int64, float64
import math


@nb.experimental.jitclass([
    ('origin', float64[:]),
    ('span', float64[:]),
    ('l0', float64),
    ('sigma', float64[:]),
    ('globalNodeIndices', int64[:])
])
class Element:
    """Base class for cuboid elements."""

    def __init__(self, origin, span, sigma):
        """
        Construct new cuboid element.

        Parameters
        ----------
        origin : float[3]
            Global Cartestian coordinates of local origin
        span : float[3]
            Size of element in each axis
        sigma : float[3] or float
            Conductivity within region
        """
        self.origin = origin
        self.span = span
        self.l0 = np.prod(span)**(1/3)
        self.sigma = sigma
        self.globalNodeIndices = np.empty(8, dtype=np.int64)

    def get_coords_recursively(self):
        """
        Calculate the global coordinates of the element's vertices.

        Returns
        -------
        coords : float[:,3]
            Cartesian coordinates of element vertices
        """
        coords = np.empty((8, 3))
        for ii in range(8):
            weights = np.array(
                [(ii >> n) & 1 for n in range(3)], dtype=np.float64)
            offset = self.origin+self.span*weights
            coords[ii] = offset

        return coords

    def get_char_length(self):
        """
        Get the characteristic length of the element.

        Returns
        -------
        l0 : float
            Characteristic length [in meters]
        """
        return math.pow(np.prod(self.span), 1.0/3)

    def setglobal_indices(self, indices):
        self.globalNodeIndices = indices


# @nb.experimental.jitclass([
#     ('origin', float64[:]),
#     ('span', float64[:]),
#     ('l0', float64),
#     ('sigma', float64[:]),
#     ('globalNodeIndices', int64[:])
# ])
# class FEMHex():
#     def __init__(self, origin, span, sigma):
#         self.origin = origin
#         self.span = span
#         self.l0 = np.prod(span)**(1/3)
#         self.sigma = sigma
#         self.globalNodeIndices = np.empty(8, dtype=np.int64)

#     def get_coords_recursively(self):
#         coords = np.empty((8, 3))
#         for ii in range(8):
#             weights = np.array(
#                 [(ii >> n) & 1 for n in range(3)], dtype=np.float64)
#             offset = self.origin+self.span*weights
#             coords[ii] = offset

#         return coords

#     def get_char_length(self):
#         return math.pow(np.prod(self.span), 1.0/3)

#     def setglobal_indices(self, indices):
#         self.globalNodeIndices = indices

#     def get_conductance_vals(self):
#         if self.sigma.shape[0] == 1:
#             sigma = self.sigma*np.ones(3)
#         else:
#             sigma = self.sigma

#         # k=self.span/(36*np.roll(self.span,1)*np.roll(self.span,2))
#         k = np.roll(self.span, 1)*np.roll(self.span, 2)/(36*self.span)
#         K = sigma*k

#         g = np.empty(28, dtype=np.float64)
#         nn = 0
#         weights = np.empty(3, dtype=np.float64)
#         for ii in range(8):
#             for jj in range(ii+1, 8):
#                 dif = np.bitwise_xor(ii, jj)

#                 mask = np.array([(dif >> i) & 1 for i in range(3)])
#                 numDif = np.sum(mask)

#                 if numDif == 1:
#                     coef = 2*(mask ^ 1)-4*mask
#                 elif numDif == 2:
#                     coef = (mask ^ 1)-2*mask
#                 else:
#                     coef = -mask

#                 weights = -coef.astype(np.float64)
#                 g0 = np.dot(K, weights)
#                 g[nn] = g0
#                 nn = nn+1

#         return g

#     def get_conductance_indices(self):
#         edges = np.empty((28, 2), dtype=np.int64)
#         nn = 0
#         for ii in range(8):
#             for jj in range(ii+1, 8):
#                 edges[nn, :] = self.globalNodeIndices[np.array([ii, jj])]
#                 nn += 1

#         return edges


# @nb.experimental.jitclass([
#     ('origin', float64[:]),
#     ('span', float64[:]),
#     ('l0', float64),
#     ('sigma', float64[:]),
#     ('globalNodeIndices', int64[:])
# ])
# class AdmittanceHex():
#     def __init__(self, origin, span, sigma):
#         self.origin = origin
#         self.span = span
#         self.l0 = np.prod(span)**(1/3)
#         self.sigma = sigma
#         self.globalNodeIndices = np.empty(8, dtype=np.int64)

#     def get_coords_recursively(self):
#         coords = np.empty((8, 3))
#         for ii in range(8):
#             weights = np.array(
#                 [(ii >> n) & 1 for n in range(3)], dtype=np.float64)
#             offset = self.origin+self.span*weights
#             coords[ii] = offset

#         return coords

#     def getMidpoint(self):
#         return self.origin+self.span/2

#     def get_char_length(self):
#         return math.pow(np.prod(self.span), 1.0/3)

#     def setglobal_indices(self, indices):
#         self.globalNodeIndices = indices

#     def get_conductance_vals(self):
#         if self.sigma.shape[0] == 1:
#             sigma = self.sigma*np.ones(3)
#         else:
#             sigma = self.sigma

#         k = np.roll(self.span, 1)*np.roll(self.span, 2)/self.span
#         K = sigma*k/4

#         g = np.array([K[ii] for ii in range(3) for jj in range(4)])

#         return g

#     def get_conductance_indices(self):
#         nodesA = np.array([0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3])
#         offsets = np.array([2**np.floor(ii/4) for ii in range(12)])
#         offsets = offsets.astype(np.int64)

#         # edges=np.array([[self.globalNodeIndices[a],self.globalNodeIndices[a+o]] for a,o in zip(nodesA,offsets)])
#         edges = np.empty((12, 2), dtype=np.int64)
#         for ii in range(12):
#             nodeA = nodesA[ii]
#             nodeB = nodeA+offsets[ii]
#             edges[ii, 0] = self.globalNodeIndices[nodeA]
#             edges[ii, 1] = self.globalNodeIndices[nodeB]
#         return edges
