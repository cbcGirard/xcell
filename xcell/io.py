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
from scipy.sparse import tril
from .geometry import fixTriNormals

import pyvista as pv
import vtk

MIO_ORDER = np.array([0, 1, 3, 2, 4, 5, 7, 6])

PV_BOUND_ORDER = np.array([0, 3, 1, 4, 2, 5])
XCELL_BOUND_ORDER = np.array([0, 2, 4, 1, 2, 5])


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
    hexInds = np.array([el.vertices[MIO_ORDER] for el in mesh.elements])
    listInds = renumberIndices(hexInds, mesh.indexMap)

    uniqueInds = np.unique(listInds.ravel())

    if mesh.nodeCoords.shape[0] != uniqueInds.shape[0]:
        pts = mesh.nodeCoords[uniqueInds]

        listInds = renumberIndices(listInds, uniqueInds)
    else:
        pts = mesh.nodeCoords

    cells = [('hexahedron', listInds)]

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
    Del = Delaunay(mesh.nodeCoords)

    surf = fixTriNormals(mesh.nodeCoords, Del.convex_hull)

    cells = [('triangle', surf)]

    mioMesh = meshio.Mesh(mesh.nodeCoords, cells)

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
    rawInd = np.array([el.vertices[MIO_ORDER] for el in mesh.elements])
    numel = rawInd.shape[0]
    trueInd = renumberIndices(rawInd, mesh.indexMap)

    cells = np.hstack(
        (8*np.ones((numel, 1), dtype=np.uint64), trueInd)).ravel()

    celltypes = np.empty(numel, dtype=np.uint8)
    celltypes[:] = vtk.VTK_HEXAHEDRON

    vMesh = pv.UnstructuredGrid(cells, celltypes, mesh.nodeCoords)

    sigmas = np.array([np.linalg.norm(el.sigma) for el in mesh.elements])
    vMesh.cell_data['sigma'] = sigmas

    return vMesh


def saveVTK(simulation, filestr):
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
    vtk = toVTK(simulation.mesh)
    vAna, _ = simulation.analyticalEstimate()
    analytic = np.sum(vAna, axis=0)
    vtk.point_data['voltage'] = simulation.nodeVoltages
    vtk.point_data['vAnalytic'] = analytic

    vtk.save(filestr)

    return vtk


def toEdgeMesh(simulation, currents=False):

    if currents:
        ivals, iedges = simulation.getEdgeCurrents()
        nEdges = iedges.shape[0]
        lines = np.hstack((2*np.ones((nEdges, 1), dtype=int),
                           iedges)).astype(int)

        gmat = tril(simulation.getEdgeMat())
        conds = np.abs(gmat.data)
    else:
        lines = np.hstack(
            (2*np.ones((simulation.edges.shape[0], 1), dtype=int), simulation.edges)).astype(int)
        conds = simulation.conductances

    edgemesh = pv.PolyData(simulation.mesh.nodeCoords,
                           faces=lines, n_faces=lines.shape[0])
    edgemesh.cell_data['Conductances'] = conds
    if currents:
        edgemesh.cell_data['Currents'] = ivals

    return edgemesh


class Regions(pv.MultiBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self['Conductors'] = pv.MultiBlock()
        self["Insulators"] = pv.MultiBlock()
        self['Electrodes'] = pv.MultiBlock()

        self.mesh = None

    def addMesh(self, mesh, category=None):
        if category is not None:
            self[category].append(mesh)
        else:
            self.mesh = mesh

    def assignSigma(self, mesh, defaultSigma=1.):
        # vtkMesh = toVTK(mesh)
        # vtkMesh.cell_data['sigma'] = defaultSigma
        # vtkPts = vtkMesh.cell_centers()
        vtkPts = pv.wrap(np.array([el.center for el in mesh.elements]))
        sigs = defaultSigma*np.ones(vtkPts.n_points)

        for region in self['Conductors']:
            sig = region['sigma'][0]
            enclosed = vtkPts.select_enclosed_points(region)
            inside = enclosed['SelectedPoints'] == 1
            # vtkMesh['sigma'][inside] = sig
            sigs[inside] = sig

        for region in self['Insulators']:
            enclosed = vtkPts.select_enclosed_points(region)
            inside = enclosed['SelectedPoints'] == 1
            # vtkMesh['sigma'][inside] = 0
            sigs[inside] = 0

        for el, s in zip(mesh.elements, sigs):  # vtkMesh['sigma']):
            el.sigma = np.array(s, ndmin=1)

        # return vtkMesh

    def toPlane(self, origin=np.zeros(3), normal=[0., 0., 1.]):
        planeRegions = Regions()
        for k in self.keys():
            for region in self[k]:
                tmp = region.slice(normal=normal,
                                   origin=origin)
                if tmp.n_points == 0:
                    tmp = region.project_points_to_plane(
                        origin=origin, normal=normal)

                planeRegions[k].append(tmp)

        return planeRegions
