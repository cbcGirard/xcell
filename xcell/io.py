#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converters for other meshing libraries
"""
import numpy as np
import numba as nb
import meshio
from .util import renumberIndices
from scipy.spatial import Delaunay
from .geometry import fixTriNormals

import pyvista as pv
import vtk

MIO_ORDER=np.array([0, 1, 3, 2, 4, 5, 7, 6])


def toMeshIO(mesh):
    """
    Format mesh for meshio.

    Parameters
    ----------
    mesh : xcell Mesh
        DESCRIPTION.

    Returns
    -------
    mioMesh : meshio Mesh
        DESCRIPTION.

    """
    hexInds=np.array([el.vertices[MIO_ORDER] for el in mesh.elements])
    listInds=renumberIndices(hexInds,mesh.indexMap)

    uniqueInds=np.unique(listInds.ravel())

    if mesh.nodeCoords.shape[0]!=uniqueInds.shape[0]:
        pts=mesh.nodeCoords[uniqueInds]

        listInds=renumberIndices(listInds,uniqueInds)
    else:
        pts=mesh.nodeCoords

    cells=[('hexahedron', listInds)]

    mioMesh = meshio.Mesh(pts, cells)

    return mioMesh

def toTriSurface(mesh):
    """
    Generate surface triangulation of mesh.

    Parameters
    ----------
    mesh : xcell Mesh
        Input xcell mesh.

    Returns
    -------
    mioMesh : meshio Mesh
        Output triangulated mesh.

    """
    Del=Delaunay(mesh.nodeCoords)

    surf=fixTriNormals(mesh.nodeCoords, Del.convex_hull)

    cells=[('triangle',surf)]

    mioMesh=meshio.Mesh(mesh.nodeCoords,cells)

    return mioMesh


def toVTK(mesh):
    """
    Export xcell mesh to a VTK Unstructured Grid.

    Enables faster visualizations and mesh operations via pyvista.

    Parameters
    ----------
    mesh : xcell Mesh
        The completed mesh.

    Returns
    -------
    vMesh : VTK Unstructured Grid
        Mesh in VTK format, for further manipulations.

    """
    rawInd=np.array([el.vertices[MIO_ORDER] for el in mesh.elements])
    numel=rawInd.shape[0]
    trueInd=renumberIndices(rawInd,mesh.indexMap)

    cells=np.hstack((8*np.ones((numel,1),dtype=np.uint64), trueInd)).ravel()

    celltypes=np.empty(numel,dtype=np.uint8)
    celltypes[:]=vtk.VTK_HEXAHEDRON

    vMesh=pv.UnstructuredGrid(cells, celltypes, mesh.nodeCoords)

    return vMesh


def saveVTK(simulation,filestr):
    """
    Save mesh in VTK format.

    Parameters
    ----------
    simulation : xcell Simulation
        DESCRIPTION.
    filestr : str
        Name of file.

    Returns
    -------
    vtk : VTK Unstructured Grid
        VTK mesh for further manipulations.

    """
    vtk=toVTK(simulation.mesh)
    vAna,_ = simulation.analyticalEstimate()
    analytic = np.sum(vAna, axis = 0)
    vtk.point_data['voltage']=simulation.nodeVoltages
    vtk.point_data['vAnalytic']=analytic

    vtk.save(filestr)

    return vtk