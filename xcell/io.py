#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:26:52 2022

format conversions for gmsh

@author: benoit




field_data:
    {'physName1':[tag,dim],
     'physName2':[tag,dim]...}

point_data:
    {'gmsh:dim_tags': [
        [entDim,entTag],
        [entDim,entTag],
        ...]}



cell_data:
    {'gmsh:physical':
     element physical group tag}






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
    Del=Delaunay(mesh.nodeCoords)

    surf=fixTriNormals(mesh.nodeCoords, Del.convex_hull)

    cells=[('triangle',surf)]

    mioMesh=meshio.Mesh(mesh.nodeCoords,cells)

    return mioMesh


def toVTK(mesh):
    rawInd=np.array([el.vertices for el in mesh.elements])
    numel=rawInd.shape[0]
    trueInd=renumberIndices(rawInd,mesh.indexMap)

    cells=np.hstack((8*np.ones((numel,1),dtype=np.uint64), trueInd)).ravel()

    celltypes=np.empty(numel,dtype=np.uint8)
    celltypes[:]=vtk.VTK_HEXAHEDRON

    vMesh=pv.UnstructuredGrid(cells, celltypes, mesh.nodeCoords)

    return vMesh