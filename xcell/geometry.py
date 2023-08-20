#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometric primitives."""

import numpy as np
import numba as nb
from numba import float64
import pyvista as pv


@nb.experimental.jitclass([("center", float64[::1]), ("radius", float64)])  # type: ignore
class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def is_inside(self, coords):
        """
        Determine which points are inside the region.

        Parameters
        ----------
        coords : float[:,3]
            Cartesian coordinates of points to test.

        Returns
        -------
        bool[:]
            Boolean array of points inside region.
        """
        N = coords.shape[0]
        isIn = np.empty(N, dtype=np.bool_)
        for ii in nb.prange(N):
            c = coords[ii]
            isIn[ii] = np.linalg.norm(c - self.center) <= self.radius

        return isIn

    def get_signed_distance(self, coords):
        N = coords.shape[0]
        dist = np.empty(N, dtype=np.float64)
        for ii in nb.prange(N):
            c = coords[ii]
            dist[ii] = np.linalg.norm(c - self.center) - self.radius

        return dist


@nb.experimental.jitclass(
    spec=[("center", float64[::1]), ("radius", float64), ("axis", float64[::1]), ("tol", float64)]
)  # type: ignore
class Disk:
    def __init__(self, center, radius, axis, tol=1e-2):
        self.center = center
        self.radius = radius
        self.axis = axis / np.linalg.norm(axis)
        self.tol = tol

    def is_inside(self, coords):
        """
        Determine which points are inside the region.

        Parameters
        ----------
        coords : float[:,3]
            Cartesian coordinates of points to test.

        Returns
        -------
        bool[:]
            Boolean array of points inside region.
        """
        N = coords.shape[0]
        isIn = np.empty(N, dtype=np.bool_)

        delta = coords - self.center
        deviation = np.dot(delta, self.axis)

        for ii in nb.prange(N):
            dist = np.linalg.norm(delta[ii])

            isIn[ii] = abs(deviation[ii]) < (self.radius * self.tol) and dist <= self.radius

        return isIn

    # todo: check math
    def get_signed_distance(self, coords):
        N = coords.shape[0]
        signedDist = np.empty(N, dtype=np.float64)

        delta = coords - self.center
        deviation = np.dot(delta, self.axis)
        # distance along axis

        for ii in nb.prange(N):
            dist = np.linalg.norm(delta[ii])
            # distance to center
            dz = deviation[ii] * self.axis - coords[ii]
            vr = dz - self.center
            dr = np.linalg.norm(vr)

            if dr < self.radius:
                # signedDist[ii] = np.linalg.norm(dz)
                signedDist[ii] = np.abs(deviation[ii])
                # projected point inside disk
            else:
                signedDist[ii] = np.sqrt(dist**2 - (dr - self.radius) ** 2)
            # else:

        return signedDist


@nb.experimental.jitclass(
    [("center", float64[::1]), ("radius", float64), ("length", float64), ("axis", float64[::1])]
)  # type: ignore
class Cylinder:
    def __init__(self, center, radius, length, axis):
        self.center = center
        self.radius = radius
        self.length = length
        self.axis = axis / np.linalg.norm(axis)

    def is_inside(self, coords):
        """
        Determine which points are inside the region.

        Parameters
        ----------
        coords : float[:,3]
            Cartesian coordinates of points to test.

        Returns
        -------
        bool[:]
            Boolean array of points inside region.
        """
        N = coords.shape[0]
        isIn = np.empty(N, dtype=np.bool_)

        delta = coords - self.center
        deviation = np.abs(np.dot(delta, self.axis))

        for ii in nb.prange(N):
            vec = delta[ii]
            dz = deviation[ii]
            dr = np.sqrt(np.dot(vec, vec) - dz**2)

            isIn[ii] = 0 <= dz <= self.length / 2 and dr <= self.radius
        return isIn


@nb.njit()
def is_in_bbox(bbox, point):
    """
    Determine whether point is within bounding box.

    Parameters
    ----------
    bbox : float[:]
        Bounding box in order (-x, -y, -z, x, y, z)
    point : float[3]
        Cartesian coordinate of point to check

    Returns
    -------
    bool
        Whether point is contained in bounding box
    """
    gt = np.greater_equal(point, bbox[:3])
    lt = np.less_equal(point, bbox[3:])

    return np.all(np.logical_and(lt, gt))


@nb.njit()
def _avg_points(pts):
    """
    Get point at average of x,y,z from each supplied point.

    Parameters
    ----------
    pts : float[:,3]
        Cartesian coordinates

    Returns
    -------
    float[:,3]
        Center of points.
    """
    center = np.zeros(3)

    for ii in nb.prange(pts.shape[0]):
        center += pts[ii, :]

    center /= pts.shape[0]

    return center


@nb.njit()
def _calc_tri_normals(pts, surf):
    ntris = surf.shape[0]
    norms = np.empty((ntris, 3))

    for ii in nb.prange(ntris):
        inds = surf[ii, :]
        a = pts[inds[0], :] - pts[inds[1], :]
        b = pts[inds[1], :] - pts[inds[2], :]

        tmp = np.cross(a, b)
        norms[ii, :] = tmp / np.linalg.norm(tmp)

    return norms


# @nb.njit()
def fix_tri_normals(pts, surf):
    """
    Experimental: try to orient all surface triangles outward.

    Parameters
    ----------
    pts : float[:,3]
        List of all vertices in surface mesh.
    surf : int[:,3]
        Indices of triangles' vertices

    Returns
    -------
    int[:,3]
        Indices of triangles' vertices, flipped as needed to orient outward.
    """
    norms = _calc_tri_normals(pts, surf)

    uniqueInds = np.unique(surf.ravel())
    ctr = _avg_points(pts[uniqueInds])

    fixedSurf = np.empty_like(surf)
    ntri = surf.shape[0]

    for ii in nb.prange(ntri):
        triCtr = _avg_points(pts[surf[ii], :])
        ok = np.dot(norms[ii, :], triCtr - ctr) > 0

        if ok:
            fixedSurf[ii, :] = surf[ii, :]
        else:
            fixedSurf[ii, :] = np.flip(surf[ii, :])

    return fixedSurf


def to_pyvista(geometry, **kwargs):
    """
    Hackishly convert xcell geometry to PyVista representation.

    Parameters
    ----------
    geometry : xcell Disk, Sphere, or Cylinder
        Geometry to convert.
    **kwargs : PyVista arguments
        Parameters for PyVista mesh generation.

    Returns
    -------
    mesh : PyVista PolyData
        Geometry as a PyVista mesh for visualization.

    """
    tstring = _get_geometry_shape(geometry)
    if tstring == "Cylinder":
        rawmesh = pv.Cylinder(
            radius=geometry.radius,
            center=geometry.center,
            height=geometry.length,
            direction=geometry.axis,
            capping=False,
            **kwargs
        )
        cells = []
        celltypes = []

        for c in rawmesh.cell:
            cells.append(c.n_points)
            cells.extend(c.point_ids)
            celltypes.append(c.type)

        npoly = rawmesh.points.shape[0] // 2

        cells.append(npoly)
        cells.extend(np.arange(0, 2 * npoly, 2, dtype=int))
        cells.append(npoly)
        cells.extend(np.arange(2 * npoly - 1, 0, -2, dtype=int))

        mesh = pv.PolyData(var_inp=rawmesh.points.copy(), faces=cells, n_faces=rawmesh.n_faces + 2)

        # tets = rawmesh.delaunay_3d()
        # mesh = tets.extract_surface()
    elif tstring == "Sphere":
        mesh = pv.Sphere(radius=geometry.radius, center=geometry.center, **kwargs)
    elif tstring == "Disk":
        mesh = pv.Disc(outer=geometry.radius, inner=0, center=geometry.center, normal=geometry.axis, **kwargs)

    return mesh


def _get_geometry_shape(geometry):
    t = str(type(geometry))
    tstring = t.split(".")[-1].split("'")[0]

    return tstring
