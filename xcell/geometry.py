#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometric primitives."""

import numpy as np
import numba as nb
from numba import int64, float64
import pyvista as pv
from .visualizers import PVScene
# from .io import toVTK


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


@nb.experimental.jitclass([
    ('center', float64[:]),
    ('radius', float64),
    ('axis', float64[:]),
    ('tol', float64)
])
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
            dist = np.linalg.norm(delta[ii])

            isIn[ii] = abs(deviation[ii]) < (
                self.radius*self.tol) and dist <= self.radius

        return isIn


@nb.experimental.jitclass([
    ('center', float64[:]),
    ('radius', float64),
    ('length', float64),
    ('axis', float64[:])
])
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
        deviation = np.abs(np.dot(delta, self.axis))

        for ii in nb.prange(N):

            vec = delta[ii]
            dz = deviation[ii]
            dr = np.sqrt(np.dot(vec, vec)-dz**2)

            isIn[ii] = 0 <= dz <= self.length/2 and dr <= self.radius
        return isIn


@nb.njit()
def isInBBox(bbox, point):
    gt = np.greater_equal(point, bbox[:3])
    lt = np.less_equal(point, bbox[3:])

    return np.all(np.logical_and(lt, gt))


@nb.njit()
def avgPoints(pts):
    center = np.zeros(3)

    for ii in nb.prange(pts.shape[0]):
        center += pts[ii, :]

    center /= pts.shape[0]

    return center


@nb.njit()
def calcTriNormals(pts, surf):
    ntris = surf.shape[0]
    norms = np.empty((ntris, 3))

    for ii in nb.prange(ntris):
        inds = surf[ii, :]
        a = pts[inds[0], :]-pts[inds[1], :]
        b = pts[inds[1], :]-pts[inds[2], :]

        tmp = np.cross(a, b)
        norms[ii, :] = tmp/np.linalg.norm(tmp)

    return norms


# @nb.njit()
def fixTriNormals(pts, surf):
    norms = calcTriNormals(pts, surf)

    uniqueInds = np.unique(surf.ravel())
    ctr = avgPoints(pts[uniqueInds])

    fixedSurf = np.empty_like(surf)
    ntri = surf.shape[0]

    for ii in nb.prange(ntri):
        triCtr = avgPoints(pts[surf[ii], :])
        ok = np.dot(norms[ii, :], triCtr-ctr) > 0

        if ok:
            fixedSurf[ii, :] = surf[ii, :]
        else:
            fixedSurf[ii, :] = np.flip(surf[ii, :])

    return fixedSurf


def toPV(geometry, **kwargs):
    t = str(type(geometry))
    tstring = t.split('.')[-1].split('\'')[0]
    if tstring == 'Cylinder':
        rawmesh = pv.Cylinder(radius=geometry.radius,
                              center=geometry.center,
                              height=geometry.length,
                              direction=geometry.axis,
                              **kwargs)
        tets = rawmesh.delaunay_3d()
        mesh = tets.extract_surface()
    elif tstring == 'Sphere':
        mesh = pv.Sphere(radius=geometry.radius,
                         center=geometry.center,
                         **kwargs)
    elif tstring == 'Disk':
        mesh = pv.Disc(outer=geometry.radius,
                       inner=0,
                       center=geometry.center,
                       normal=geometry.axis,
                       **kwargs)

    return mesh
