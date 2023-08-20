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


@nb.experimental.jitclass(
    [
        ("origin", float64[:]),
        ("span", float64[:]),
        ("l0", float64),
        ("sigma", float64[:]),
        ("globalNodeIndices", int64[:]),
    ]
)
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
        self.l0 = np.prod(span) ** (1 / 3)
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
            weights = np.array([(ii >> n) & 1 for n in range(3)], dtype=np.float64)
            offset = self.origin + self.span * weights
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
        return math.pow(np.prod(self.span), 1.0 / 3)

    def setglobal_indices(self, indices):
        self.globalNodeIndices = indices
