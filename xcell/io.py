#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converters for other meshing libraries
"""
import numpy as np
import numba as nb
import meshio
from .util import renumber_indices
from scipy.spatial import Delaunay
from scipy.sparse import tril
from .geometry import fix_tri_normals

import pyvista as pv
import vtk

#: Access xcell vertices according to meshio's ordering
TO_MIO_VERTEX_ORDER = np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=int)
#: Access meshio vertices according to xcell's ordering
FROM_MIO_VERTEX_ORDER = np.argsort(TO_MIO_VERTEX_ORDER)

TO_PV_BBOX_ORDER = np.array([0, 3, 1, 4, 2, 5])
FROM_PV_BBOX_ORDER = np.array([0, 2, 4, 1, 2, 5])


def to_meshio(mesh):
    """
    Format mesh for meshio.

    Parameters
    ----------
    mesh : xcell Mesh
        Mesh to convert.

    Returns
    -------
    mioMesh : meshio Mesh
        Converted mesh.

    """
    hexInds = np.array([el.vertices[TO_MIO_VERTEX_ORDER] for el in mesh.elements])
    listInds = renumber_indices(hexInds, mesh.index_map)

    uniqueInds = np.unique(listInds.ravel())

    if mesh.node_coords.shape[0] != uniqueInds.shape[0]:
        pts = mesh.node_coords[uniqueInds]

        listInds = renumber_indices(listInds, uniqueInds)
    else:
        pts = mesh.node_coords

    cells = [("hexahedron", listInds)]

    mioMesh = meshio.Mesh(pts, cells)

    return mioMesh


# TODO: deprecate in favor of Pyvista routines?
# def toTriSurface(mesh):
#     """
#     Generate surface triangulation of mesh.

#     Parameters
#     ----------
#     mesh : xcell Mesh
#         Input xcell mesh.

#     Returns
#     -------
#     mioMesh : meshio Mesh
#         Output triangulated mesh.

#     """
#     Del = Delaunay(mesh.node_coords)

#     surf = fix_tri_normals(mesh.node_coords, Del.convex_hull)

#     cells = [('triangle', surf)]

#     mioMesh = meshio.Mesh(mesh.node_coords, cells)

#     return mioMesh


def to_vtk(mesh):
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
    rawInd = np.array([el.vertices[TO_MIO_VERTEX_ORDER] for el in mesh.elements])
    numel = rawInd.shape[0]
    trueInd = renumber_indices(rawInd, mesh.index_map)

    cells = np.hstack((8 * np.ones((numel, 1), dtype=np.uint64), trueInd)).ravel()

    celltypes = np.empty(numel, dtype=np.uint8)
    celltypes[:] = vtk.VTK_HEXAHEDRON

    vMesh = pv.UnstructuredGrid(cells, celltypes, mesh.node_coords)

    sigmas = np.array([np.linalg.norm(el.sigma) for el in mesh.elements])
    vMesh.cell_data["sigma"] = sigmas

    return vMesh


def save_vtk(simulation, filestr):
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
    vtk = to_vtk(simulation.mesh)
    vAna, _ = simulation.calculate_analytical_voltage()
    analytic = np.sum(vAna, axis=0)
    vtk.point_data["voltage"] = simulation.node_voltages
    vtk.point_data["vAnalytic"] = analytic

    vtk.save(filestr)

    return vtk


class Regions(pv.MultiBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self["Conductors"] = pv.MultiBlock()
        self["Insulators"] = pv.MultiBlock()
        self["Electrodes"] = pv.MultiBlock()

        self.sim_mesh = None

    def addMesh(self, mesh, category=None):
        if category is not None:
            self[category].append(mesh)
        else:
            self.sim_mesh = mesh

    def assign_sigma(self, sim_mesh, default_sigma=1.0):
        """
        Set the conductivity of each cell in sim_mesh based on region geometry.

        Parameters
        ----------
        mesh : Pyvista or Xcell mesh
            Domain-spanning mesh to set sigma of
        default_sigma : float, optional
            Default, by default 1.0
        """
        isPV = isinstance(sim_mesh, pv.DataSet)
        if isPV:
            vtkPts = pv.wrap(sim_mesh.cell_centers())
        else:
            vtkPts = pv.wrap(np.array([el.center for el in sim_mesh.elements]))

        sigs = default_sigma * np.ones(vtkPts.n_points)

        for region in self["Conductors"]:
            sig = region["sigma"][0]
            enclosed = vtkPts.select_enclosed_points(region)
            inside = enclosed["SelectedPoints"] == 1
            # vtkMesh['sigma'][inside] = sig
            sigs[inside] = sig

        for region in self["Insulators"]:
            enclosed = vtkPts.select_enclosed_points(region)
            inside = enclosed["SelectedPoints"] == 1
            # vtkMesh['sigma'][inside] = 0
            sigs[inside] = 0

        if isPV:
            sim_mesh.cell_data["sigma"] = sigs.copy()
        else:
            for el, s in zip(sim_mesh.elements, sigs):  # vtkMesh['sigma']):
                el.sigma = np.array(s, ndmin=1)

        # return vtkMesh

    def toPlane(self, origin=np.zeros(3), normal=[0.0, 0.0, 1.0]):
        planeRegions = Regions()
        for k in self.keys():
            for region in self[k]:
                tmp = region.slice(normal=normal, origin=origin)
                if tmp.n_points == 0:
                    tmp = region.project_points_to_plane(origin=origin, normal=normal)

                planeRegions[k].append(tmp)

        return planeRegions
