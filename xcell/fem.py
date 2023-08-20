#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematics for discrete conductivities within elements.
"""

import numpy as np
import numba as nb
from . import util


@nb.njit()
def get_hex_conductances(span, sigma):
    """
    Calculate the conductances of a trilinear FEM hexahedron.

    Parameters
    ----------
    span : float[3]
        Size of element along each axis
    sigma : float[3] or float
        Conductivity of element

    Returns
    -------
    g : float[28]
        Conductances of element between each nodal pair in :py:const:`HEX_EDGES`

    """
    if sigma.shape[0] == 1:
        sigma = sigma * np.ones(3)
    else:
        sigma = sigma

    k = np.roll(span, 1) * np.roll(span, 2) / (36 * span)
    K = sigma * k

    g = np.empty(28, dtype=np.float64)
    nn = 0
    weights = np.empty(3, dtype=np.float64)
    for ii in range(8):
        for jj in range(ii + 1, 8):
            dif = np.bitwise_xor(ii, jj)

            mask = np.array([(dif >> i) & 1 for i in range(3)])
            numDif = np.sum(mask)

            if numDif == 1:
                coef = 2 * (mask ^ 1) - 4 * mask
            elif numDif == 2:
                coef = (mask ^ 1) - 2 * mask
            else:
                coef = -mask

            weights = -coef.astype(np.float64)
            g0 = np.dot(K, weights)
            g[nn] = g0
            nn = nn + 1

    return g


@nb.njit()
def _gen_hex_indices():
    """Generate indices of conductance nodes for trilinear FEM hexahedron.

    Returns
    -------
    int64[28,2]
        Edges within the hex
    """
    edges = np.empty((28, 2), dtype=np.int64)
    nn = 0
    for ii in range(8):
        for jj in range(ii + 1, 8):
            edges[nn, :] = np.array([ii, jj])
            nn += 1

    return edges


@nb.njit()
def get_face_conductances(span, sigma):
    """
    Calculate the conductances of a mesh-dual (face-oriented) hexahedron.

    Parameters
    ----------
    span : float[3]
        Size of element along each axis
    sigma : float[3] or float
        Conductivity of element

    Returns
    -------
    float[6]
        Conductances, in order of :py:const:`FACE_EDGES`{-x, +x, -y, +y, ...}.

    """
    if sigma.shape[0] == 1:
        sigma = sigma * np.ones(3)
    else:
        sigma = sigma

    k = np.roll(span, 1) * np.roll(span, 2) / span
    g = sigma * k * 2
    return g.repeat(2)


@nb.njit()
def _gen_face_indices():
    subsets = np.array([[0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6]], dtype=np.int64)
    return subsets


@nb.njit()
def get_tet_conductance(pts):
    """
    Calculate conductances between nodes of tetrahedron.

    Parameters
    ----------
    pts : float[4,3]
        Cartesian coordinates of tet vertices

    Returns
    -------
    float[4]
        Conductances between vertices in order of :py:const:`TET_EDGES`
    """
    g = np.empty(6, dtype=np.float64)
    for ii in range(6):
        edge = TET_EDGES[ii]
        others = TET_EDGES[5 - ii]
        otherPts = pts[others]

        M = np.vstack((otherPts[1], pts[edge[0]], pts[edge[1]])) - otherPts[0]
        A = M[0, :]
        B = M[1, :]
        C = M[2, :]

        num = np.dot(A, B) * np.dot(A, C) - np.dot(A, A) * np.dot(B, C)
        den = np.abs(6 * np.linalg.det(M))
        g[ii] = -num / den

    return g


@nb.njit()
def _gen_tet_indices():
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    return edges


@nb.njit()
def _get_admittance_conductances(span, sigma):
    if sigma.shape[0] == 1:
        sigma = sigma * np.ones(3)
    else:
        sigma = sigma

    k = np.roll(span, 1) * np.roll(span, 2) / span
    K = sigma * k / 4

    g = np.array([K[ii] for ii in range(3) for jj in range(4)])

    return g


@nb.njit()
def _gen_admittance_indices():
    nodesA = np.array([0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3], dtype=np.int64)
    offsets = np.array([2 ** np.floor(ii / 4) for ii in range(12)], dtype=np.int64)

    edges = np.vstack((nodesA, nodesA + offsets)).transpose()
    return edges


#: Indices of conductances within admittance element
ADMITTANCE_EDGES = _gen_admittance_indices()
#: Indices of conductances within tetrahedral element
TET_EDGES = _gen_tet_indices()
#: Indices of nodes connected by edge mesh-dual element
FACE_EDGES = _gen_face_indices()
#: Indices of nodes connected by edge in hex element
HEX_EDGES = _gen_hex_indices()


@nb.njit()
def interpolate_from_face(faceValues, localCoords):
    """
    Interpolate within the cuboid from values on the faces.

    Parameters
    ----------
    faceValues : float[:]
        Known values at each face.
    localCoords : float[:,3]
        Cartesian coordinates at which to interpolate.

    Returns
    -------
    float[:]
        Interpolated values.
    """

    # coef=_TRIPOLAR_INVERSE_MATRIX @ faceValues
    coef = np.dot(_TRIPOLAR_INVERSE_MATRIX, faceValues)
    cvals = __toFaceCoefOrder(localCoords)

    interpv = np.dot(cvals, coef)
    return interpv


@nb.njit()
def interpolate_from_verts(vertexValues, localCoords):
    """
    Interpolate within the element from values on the vertices (i.e. trilinear interpolation).

    Parameters
    ----------
    vertexValues : float[:]
        Known values at each vertex.
    localCoords : float[:,3]
        Cartesian coordinates at which to interpolate.

    Returns
    -------
    float[:]
        Interpolated values.
    """
    # coef=_TRILIN_INVERSE_MATRIX @ vertexValues
    coef = np.dot(_TRILIN_INVERSE_MATRIX, vertexValues)
    cvals = _to_trilinear_coefficent_order(localCoords)

    interpV = np.dot(cvals, coef)
    return interpV


@nb.njit()
def integrate_from_verts(vertexValues, span):
    """
    Integrate values on the cube assuming trilinear interpolation.

    Parameters
    ----------
    vertexValues : float[:]
        Values at vertices
    span : float[3]
        Length of cube in x,y,z directions

    Returns
    -------
    float
        Integral of input values over the cube's volume.
    """

    vals = np.dot(_TRILIN_INVERSE_MATRIX, vertexValues)
    xyz = np.prod(span)
    coef = _to_trilinear_coefficent_order(np.array(span, ndmin=2)) * xyz / np.array([1, 2, 2, 2, 4, 4, 4, 8])

    integral = np.dot(coef, vals)

    return integral


#     Vxyz =	V000 (1 - x) (1 - y) (1 - z) +
# V100 x (1 - y) (1 - z) +
# V010 (1 - x) y (1 - z) +
# V001 (1 - x) (1 - y) z +
# V101 x (1 - y) z +
# V011 (1 - x) y z +
# V110 x y (1 - z) +
# V111 x y z


@nb.njit()
def to_local_coords(global_coords, center, span):
    """
    Transform global Cartesian coordinates to an element's local coordinate system (-1,1,-1...).

    Parameters
    ----------
    global_coords : float[:,3]
        Global cartesian coordinates
    center : float[3]
        Center of cube in global coordinates.
    span : float[3]
        Length of cube in x,y,z.

    Returns
    -------
    float[:,3]
        Points denoted in the element's local coordinate system.
    """
    localCoords = np.empty_like(global_coords)

    for ii in nb.prange(global_coords.shape[0]):
        localCoords[ii] = 2 * (global_coords[ii] - center) / span

    return localCoords


@nb.njit()
def _to_trilinear_coefficent_order(coords):
    npts = coords.shape[0]
    ordered = np.empty((npts, 8), dtype=np.float64)
    for ii in nb.prange(npts):
        c = coords[ii, :]
        ordered[ii, 0] = 1
        ordered[ii, 1:4] = c
        ordered[ii, 4] = c[0] * c[1]
        ordered[ii, 5] = c[0] * c[2]
        ordered[ii, 6] = c[1] * c[2]
        ordered[ii, 7] = np.prod(c)

    return ordered


@nb.njit()
def __toFaceCoefOrder(coords):
    npts = coords.shape[0]
    ordered = np.empty((npts, 7))
    for ii in nb.prange(npts):
        ordered[ii, :3] = coords[ii]
        ordered[ii, 3:6] = coords[ii] ** 2

    ordered[:, 6] = 1
    return ordered


#: Local coordinates of vertices for interpolation
HEX_VERTEX_COORDS = np.array(
    [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
    dtype=np.float64,
)

#: Local coordinates of faces for interpolation
HEX_FACE_COORDS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [0, 0, 0]], dtype=np.float64
)

#: Universal points within the cubic element
HEX_POINT_INDICES = np.vstack((1 + HEX_VERTEX_COORDS, 1 + HEX_FACE_COORDS)).astype(np.int32)

_TRILIN_INVERSE_MATRIX = np.linalg.inv(_to_trilinear_coefficent_order(HEX_VERTEX_COORDS))

_TRIPOLAR_INVERSE_MATRIX = np.linalg.inv(__toFaceCoefOrder(HEX_FACE_COORDS))
