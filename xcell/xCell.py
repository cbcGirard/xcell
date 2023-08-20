#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main API for handling extracellular simulations."""

from sre_parse import State
from turtle import st
import numpy as np
import numba as nb

from numba import int64, float64
import math
import scipy
from scipy.sparse.linalg import spsolve, cg

# from visualizers import *
# from util import *
import os
import pickle

import matplotlib.ticker as tickr
import matplotlib.pyplot as plt

# plt.style.use('dark_background')


from . import util
from . import visualizers
from . import elements
from . import meshes
from . import geometry
from .fem import ADMITTANCE_EDGES
from . import misc
from . import colors
from .signals import Signal


# TODO: test njit of source classes
# @nb.experimental.jitclass([
#     ('value',float64),
#     ('coords',float64[:]),
#     ('radius',float64)
#      ])
class Source:
    def __init__(self, value, geom):
        self.value = value
        self.geometry = geom

    def __getstate__(self):
        state = self.__dict__.copy()

        shape = geometry._get_geometry_shape(self.geometry)
        geodict = {"shape": shape}
        geodict["center"] = self.geometry.center
        geodict["radius"] = self.geometry.radius

        if shape == "Disk":
            geodict["axis"] = self.geometry.axis
            geodict["tol"] = self.geometry.tol
        if shape == "Cylinder":
            geodict["length"] = self.geometry.length
            geodict["axis"] = self.geometry.axis

        state["geometry"] = geodict

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        geodict = self.geometry.copy()

        shape = geodict["shape"]
        if shape == "Sphere":
            self.geometry = geometry.Sphere(center=geodict["center"], radius=geodict["radius"])
        if shape == "Disk":
            self.geometry = geometry.Disk(
                center=geodict["center"], radius=geodict["radius"], axis=geodict["axis"], tol=geodict["tol"]
            )
        if shape == "Cylinder":
            self.geometry = geometry.Cylinder(
                center=geodict["center"], radius=geodict["radius"], length=geodict["length"], axis=geodict["axis"]
            )


class Simulation:
    def __init__(self, name, bbox, print_step_times=False):
        """
        Create new simulation

        Parameters
        ----------
        name : string
            Name of simulation, e.g. for file naming
        bbox : float[6]
            _description_
        print_step_times : bool, optional
            Output step timing info, by default False
        """
        self.current_sources = []
        self.voltage_sources = []

        # self.vSourceNodes = []
        # self.vSourceVals = []

        self.node_role_table = np.empty(0)
        self.node_role_values = np.empty(0)

        self.mesh = meshes.Mesh(bbox)
        self.current_time = 0.0
        self.iteration = 0
        self.meshnum = 0

        self.step_logs = []
        self.memUsage = 0
        self.print_times = print_step_times

        self.node_voltages = np.empty(0)
        self.edges = [[]]

        self.gMat = []
        self.RHS = []
        self.nDoF = 0

        self.name = name
        self.meshtype = "uniform"

        self.ptPerAxis = 0

        self.asDual = False

    def quick_adaptive_grid(self, max_depth, coefficent=0.2):
        """
        Make a generic octree mesh where resolution increases near
        current sources.

        Parameters
        ----------
        max_depth : int
            Maximum subdivision depth.
        coefficent : float
            Density term (maximum resolution in whole mesh at 1.0)

        Returns
        -------
        None.

        """
        pts = np.array([a.geometry.center for a in self.current_sources])

        npt = pts.shape[0]

        self.make_adaptive_grid(
            ref_pts=pts,
            max_depth=max_depth * np.ones(npt),
            min_l0_function=general_metric,
            coefs=coefficent * np.ones(npt),
            coarsen=False,
        )

        self.finalize_mesh()

    def make_adaptive_grid(
        self, ref_pts, max_depth, min_l0_function, max_l0_function=None, coefs=None, coarsen=True
    ):
        """
        Construct octree mesh.

        Parameters
        ----------
        ref_pts : float[:,3]
            DESCRIPTION.
        max_depth : int
            Maximum recursion depth.
        min_l0_function : function
            Function to calculate l0. cf :meth:`xcell.general_metric`.
        max_l0_function : function, optional
            DESCRIPTION. min_l0_function used if None.
        coefs : float[:], optional
            Coefficents passed to l0 function. The default is None.
        coarsen : bool, optional
            Whether to prune following splits. The default is True.

        Returns
        -------
        changed : bool
            Whether adaptation results in a different mesh topology

        """
        self.start_timing("Make elements")
        self.ptPerAxis = 2**max_depth + 1
        self.meshtype = "adaptive"

        if coefs is None:
            coefs = np.ones(ref_pts.shape[0])

        # convert to octree mesh
        if type(self.mesh) != meshes.Octree:
            bbox = self.mesh.bbox

            self.mesh = meshes.Octree(bbox, max_depth, element_type=self.mesh.element_type)

        self.mesh.max_depth = max_depth

        changed = self.mesh.refine_by_metric(min_l0_function, ref_pts, max_l0_function, coefs, coarsen=coarsen)
        self.log_time()

        return changed

    def make_uniform_grid(self, nX, sigma=np.array([1.0, 1.0, 1.0])):
        """
        Fast utility to construct a uniformly spaced mesh of the domain.

        Parameters
        ----------
        nX : int
            Number of elements along each axis
            (yielding nX**3 total elements).
        sigma : float or float[3], optional
            Global conductivity. The default is np.array([1.,1.,1.]).

        Returns
        -------
        None.

        """
        self.meshtype = "uniform"
        self.start_timing("Make elements")

        xmax = self.mesh.extents[0]
        self.ptPerAxis = nX + 1

        xx = np.linspace(self.mesh.bbox[0], self.mesh.bbox[3], nX + 1)
        yy = np.linspace(self.mesh.bbox[1], self.mesh.bbox[4], nX + 1)
        zz = np.linspace(self.mesh.bbox[2], self.mesh.bbox[5], nX + 1)
        XX, YY, ZZ = np.meshgrid(xx, yy, zz)

        coords = np.vstack((XX.ravel(), YY.ravel(), ZZ.ravel())).transpose()

        self.mesh.node_coords = coords

        elOffsets = np.array([1, nX + 1, (nX + 1) ** 2])
        nodeOffsets = np.array([np.dot(util.to_bit_array(i), elOffsets) for i in range(8)])
        elExtents = self.mesh.span / nX

        for zz in range(nX):
            for yy in range(nX):
                for xx in range(nX):
                    elOriginNode = xx + yy * (nX + 1) + zz * (nX + 1) ** 2
                    origin = coords[elOriginNode]
                    elementNodes = elOriginNode + nodeOffsets

                    self.mesh.add_element(origin, elExtents, sigma, elementNodes)
                    self.mesh.elements[-1].vertices = elementNodes

        self.log_time()
        print("%d elements in mesh" % (nX**3))

    def start_timing(self, step_name):
        """
        General call to start timing an execution step.

        Parameters
        ----------
        step_name : string
            Label for the step

        Returns
        -------
        None.

        """
        logger = util.Logger(step_name, self.print_times)
        self.step_logs.append(logger)

    def log_time(self, logger=None):
        """
        Signals completion of step.

        Parameters
        ----------
        logger : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if logger is None:
            logger = self.step_logs[-1]
        logger.logCompletion()

    def getMemUsage(self, print_value=False):
        """
        Get memory usage of simulation.

        Parameters
        ----------
        print_value : boolean, optional
            Print current memory usage

        Returns
        -------
        mem : int
            Platform-dependent, often kb used.

        """
        mem = 0
        for log in self.step_logs:
            mem = max(mem, log.memory)

        if print_value:
            engFormat = tickr.EngFormatter(unit="b")
            print(engFormat(mem) + " used")

        return mem

    def print_total_time(self):
        """Print total simulation time"""
        tCPU = 0
        tWall = 0
        for l in self.step_logs:
            tCPU += l.durationCPU
            tWall += l.durationWall

        engFormat = tickr.EngFormatter()
        print("\tTotal time: " + engFormat(tCPU) + "s [CPU], " + engFormat(tWall) + "s [Wall]")

    def get_edge_currents(self):
        """
        Get currents through each edge of the mesh.

        Returns
        -------
        currents : float[:]
            Current through edge in amperes; .
        edges : int[:,:]
            Pairs of node (global) node indices corresponding to
            [start, end] of each current vector.

        """
        gAll = self.get_edge_matrix()
        condMat = scipy.sparse.tril(gAll, -1)
        edges = np.array(condMat.nonzero()).transpose()

        dv = np.diff(self.node_voltages[edges]).squeeze()
        iTmp = -condMat.data * dv

        # make currents positive, and flip direction if negative
        needsFlip = iTmp < 0
        currents = abs(iTmp)
        edges[needsFlip] = np.fliplr(edges[needsFlip])

        return currents, edges

    def intify_coords(self, coords=None):
        """
        Expresses coordinates as triplet of positive integers.

        Prevents rounding errors when determining if two points correspond
        to the same mesh node

        Parameters
        ----------
        coords: float[:,:]
            Coordinates to rescale as integers, or mesh nodes if None.

        Returns
        -------
        int[:,:]
            Mesh nodes as integers.

        """
        nx = self.ptPerAxis - 1
        bb = self.mesh.bbox

        if coords is None:
            coords = self.mesh.node_coords

        span = bb[3:] - bb[:3]
        float0 = coords - bb[:3]
        ints = np.rint((nx * float0) / span)

        return ints.astype(np.int64)

    def _make_table_header(self):
        cols = [
            "File name",
            "Mesh type",
            "Domain size",
            "Element type",
            "Number of nodes",
            "Number of elements",
        ]

        for time_type in ["CPU", "Wall"]:
            for log in self.step_logs:
                cols.append(log.name + " [" + time_type + "]")

        cols.extend(["Total time [CPU]", "Total time [Wall]", "Max memory"])
        return ",".join(cols)

    def log_table_entry(self, csvFile, extraCols=None, extraVals=None):
        """
        Log key metrics of simulation as an additional line of a .csv file.

        Custom categories (column headers) and their values can be added
        to the line.

        Parameters
        ----------
        csvFile : file path
            File where data is written to.
        extraCols : string[:], optional
            Additional categories (column headers). The default is None.
        extraVals : numeric[:], optional
            Values corresponding to the additional categories.
            The default is None.

        Returns
        -------
        None.

        """
        oldfile = os.path.exists(csvFile)
        f = open(csvFile, "a")

        if not oldfile:
            f.write(self._make_table_header())

            if extraCols is not None:
                f.write("," + ",".join(extraCols))

            f.write("\n")

        data = [
            self.name,
            self.meshtype,
            np.mean(self.mesh.extents),
            self.mesh.element_type,
            self.mesh.node_coords.shape[0],
            len(self.mesh.elements),
        ]

        memory = 0
        cpuTimes = []
        wallTimes = []

        for log in self.step_logs:
            cpuTimes.append(log.durationCPU)
            wallTimes.append(log.durationWall)
            memory = max(memory, log.memory)

        data.extend(cpuTimes)
        data.extend(wallTimes)

        data.append(sum(cpuTimes))
        data.append(sum(wallTimes))

        f.write(",".join(map(str, data)))
        f.write("," + str(memory))
        if extraVals is not None:
            f.write("," + ",".join(map(str, extraVals)))

        f.write("\n")
        f.close()

    def finalize_mesh(self, regularize=False, sigma_mesh=None, default_sigma=1.0):
        """
        Prepare mesh for simulation.

        Locks connectivity, sets global node numbering, gets list of
        edges and corresponding conductances from all elements.

        Parameters
        ----------
        regularize : bool, optional
            Perform connectivity regularization, by default False
            Not currently implemented
        sigma_mesh : mesh, optional
            Mesh defining conductivity values, by default None
        default_sigma : float, optional
            Fallback conductivity for elements outside sigma_mesh,
            by default 1.0.
        """
        self.start_timing("Finalize mesh")
        self.mesh.finalize()

        #: Sum of currents into each node with >=1 source attached
        self._node_current_sources = []
        self.log_time()

        if sigma_mesh is not None:
            self.start_timing("Assign sigma")

            vmesh = sigma_mesh.assign_sigma(self.mesh, default_sigma=default_sigma)
            self.log_time()

        self.start_timing("Calculate conductances")
        edges, conductances, transforms = self.mesh.getConductances()
        # self.edges=edges
        self.conductances = conductances
        self.log_time()

        self.start_timing("Renumber nodes")
        connInds = np.unique(edges).astype(np.uint64)
        floatInds = np.array([xf[-1] for xf in transforms], dtype=np.uint64)
        allInds = np.concatenate((connInds, floatInds), dtype=np.uint64)

        # label points lacking a corresponding edge (floaters in transform)
        # okPts=np.isin(self.mesh.index_map,connInds,assume_unique=True)
        self.transforms = transforms

        # # Explicit exclusion of unconnected nodes
        # self.mesh.index_map=connInds
        # self.mesh.inverse_index_map=util.get_index_dict(connInds)
        # self.mesh.node_coords=util.indices_to_coordinates(connInds,
        #                                         self.mesh.bbox[:3],
        #                                         self.mesh.span)

        self.mesh.index_map = allInds
        idic = util.get_py_dict(allInds)
        self.mesh.inverse_index_map = idic

        if self.meshtype != "uniform":
            self.mesh.node_coords = util.indices_to_coordinates(allInds, self.mesh.bbox[:3], self.mesh.span)

            self.edges = util.renumber_indices(edges, self.mesh.index_map)
        else:
            self.edges = edges
        nNodes = self.mesh.node_coords.shape[0]

        self.node_role_table = np.zeros(nNodes, dtype=np.int64)
        self.node_role_values = np.zeros(nNodes, dtype=np.int64)

        # tag floaters from exclusion in system of equations
        # self.node_role_table[connInds.shape[0]:]=-1

        self.log_time()

    def apply_transforms(self):
        """
        Calculate implicitly defined node voltages from explicit ones.

        Returns
        -------
        None.

        """
        targetPts = [self.mesh.inverse_index_map[xf.pop()] for xf in self.transforms]

        for ii, xf in zip(targetPts, self.transforms):
            pts = np.array([self.mesh.inverse_index_map[nn] for nn in xf])
            self.node_voltages[ii] = np.mean(self.node_voltages[pts])

    def add_current_source(self, value, geometry):
        """
        Add current source to setup.

        Parameters
        ----------
        value : float or :py:class:`xcell.signals.Signal`
            Amplitude of source
        geometry : :py:class:`xcell.geometry.Shape`

        """
        src = Source(value, geometry)

        self.current_sources.append(src)

    def add_voltage_source(self, value, geometry):
        """
        Add voltage source to setup.

        Parameters
        ----------
        value : float or :py:class:`xcell.signals.Signal`
            Amplitude of source
        geometry : :py:class:`xcell.geometry.Shape`

        """
        self.voltage_sources.append(Source(value, geometry))

    def insert_sources_in_mesh(self, snaplength=0.0):
        """_summary_

        Parameters
        ----------
        snaplength : float, optional
            Maximum distance allowed between source and mesh node,
            by default 0.0.
            Value currently unused.
        """
        for ii in nb.prange(len(self.voltage_sources)):
            src = self.voltage_sources[ii]

            indices = self.__nodesInSource(src)

            self.node_role_table[indices] = 1
            self.node_role_values[indices] = ii

            # self.node_voltages[indices] = src.value
            self.node_voltages[indices] = src.value.get_value_at_time(self.current_time)

        # meshCurrentSrc=self._node_current_sources
        # meshCurrentSrc=[0 for k in self._node_current_sources]
        meshCurrentSrc = []

        for ii in nb.prange(len(self.current_sources)):
            src = self.current_sources[ii]
            currentValue = src.value.get_value_at_time(self.current_time)

            ######
            # #TODO: introduces bugs?
            # if 'geometry' in dir(src):
            #     indices = np.nonzero(
            #         src.geometry.is_inside(self.mesh.node_coords))[0]
            # else:

            indices = self.__nodesInSource(src)

            indNodeSrc = len(meshCurrentSrc)
            for ni, idx in enumerate(indices):
                # #TODO: rollback to fix single source
                # self.node_role_table[idx]=2
                # self.node_role_values[idx]=ii

                # scenarios
                # node is one of many in source
                # node is lone of several sources
                # node is lone of source

                if self.node_role_table[idx] == 2:
                    # node is already claimed by source
                    sharedIdx = self.node_role_values[idx]
                    # add to value of existing node source
                    if sharedIdx >= len(meshCurrentSrc):
                        # TODO: Is this a bugfix, or just dumb?
                        # print('')
                        meshCurrentSrc.append(currentValue)
                    else:
                        if ii != sharedIdx:
                            meshCurrentSrc[sharedIdx] += currentValue
                    # src.value
                else:
                    # normal, set as current dof
                    self.node_role_table[idx] = 2

                    if ni == 0:
                        meshCurrentSrc.append(currentValue)
                        # .src.value)

                    self.node_role_values[idx] = indNodeSrc

        self._node_current_sources = meshCurrentSrc

    def __nodesInSource(self, source):
        if "geometry" in dir(source):
            source_center = source.geometry.center
            inside = source.geometry.is_inside(self.mesh.node_coords)
        else:
            source_center = source.coords
            d = np.linalg.norm(source_center - self.mesh.node_coords, axis=1)
            inside = d <= source.radius

        if sum(inside) > 0:
            # Grab all nodes inside of source
            index = np.nonzero(inside)[0]

        else:
            # Get closest mesh node
            el = self.mesh.get_containing_element(source_center)

            if self.mesh.element_type == "Face":
                element_universal_indicess = el.faces
            else:
                element_universal_indicess = el.vertices

            elIndices = np.array([self.mesh.inverse_index_map[n] for n in element_universal_indicess])
            elCoords = self.mesh.node_coords[elIndices]

            d = np.linalg.norm(source.geometry.center - elCoords, axis=1)
            index = elIndices[d == min(d)]

        return index

    def set_boundary_nodes(self, boundary_function=None, expand=False, sigma=1.0):
        """
        Set potential of nodes at simulation boundary.

        Can pass user-defined function to calculate node potential from its
        Cartesian coordinates; otherwise, boundary is grounded.

        Parameters
        ----------
        boundary_function : function, optional
            User-defined potential as function of coords.
            The default is None.
        expand : bool, optional
            Embed current domain as center of 3x3x3 grid of cubes before
            assigning boundaries.
        sigma : float, optional
            Conductivity to assign the new elements if expand is True.
            Default is 1.0.

        Returns
        -------
        None.

        """
        bnodes = self.mesh.get_boundary_nodes()

        self.node_voltages = np.zeros(self.mesh.node_coords.shape[0])

        if not expand:
            if boundary_function is None:
                bvals = np.zeros_like(bnodes)
            else:
                bcoords = self.mesh.node_coords[bnodes]
                blist = []
                for ii in nb.prange(len(bnodes)):
                    blist.append(boundary_function(bcoords[ii]))
                bvals = np.array(blist)

            self.node_voltages[bnodes] = bvals
            self.node_role_table[bnodes] = 1
        else:
            oldRoles = self.node_role_table
            oldEdges = self.edges
            oldConductances = self.conductances
            nInt = oldRoles.shape[0]

            # v0: only corners
            nX = util.MAXPT
            xyz = np.array(
                [[x, y, z] for z in [0, nX - 1] for y in [0, nX - 1] for x in [0, nX - 1]], dtype=np.uint64
            )
            ind = util.position_to_index(xyz, nX)
            bnodes = np.array([self.mesh.inverse_index_map[n] for n in ind])

            new_edges = np.array([[n, nInt] for n in bnodes], dtype=np.uint64)

            # TODO: assumes isotropic mesh/conductance
            cond = sum(sigma * self.mesh.span)

            self.node_role_values = np.concatenate((self.node_role_values, [0.0]))
            self.node_role_table = np.concatenate((oldRoles, [1]))
            self.node_voltages = np.zeros(oldRoles.shape[0] + 1)

            self.edges = np.vstack((oldEdges, new_edges))
            self.conductances = np.concatenate((oldConductances, cond * np.ones(bnodes.shape)))

    def solve(self, iterative=True, tol=1e-12, v_guess=None):
        """
        Directly solve for nodal voltages.

        Computational time grows significantly with simulation size;
        try solve() for faster convergence

        Returns
        -------
        voltages : float[:]
            Simulated nodal voltages.

        See Also
        --------
        solve: conjugate gradient solver

        """
        self.start_timing("Sort node types")
        self._get_node_types()
        self.log_time()

        # nNodes=self.mesh.node_coords.shape[0]
        voltages = self.node_voltages

        dof2Global = np.nonzero(self.node_role_table == 0)[0]
        nDoF = dof2Global.shape[0]

        M, b = self._construct_system()

        self.start_timing("Solve")

        if iterative:
            vDoF, cginfo = cg(M.tocsc(), b, v_guess, tol)
            self.log_time()

            if cginfo != 0:
                print("cg:%d" % cginfo)
        else:
            # Direct sparse solver
            vDoF = spsolve(M.tocsc(), b)
            self.log_time()

        voltages[dof2Global] = vDoF[:nDoF]

        for nn in range(nDoF, len(vDoF)):
            sel = self._select_by_dof(nn)
            voltages[sel] = vDoF[nn]

        self.node_voltages = voltages
        return voltages

    def get_voltage_at_dof(self):
        """
        Get the voltage of every degree of freedom.

        Returns
        -------
        vDoF : float[:]
            Voltages of all degrees of freedom [floating nodes + current sources].

        """
        isDoF = self.node_role_table == 0
        ndof = util.fastcount(isDoF)

        nsrc = len(self._node_current_sources)

        vDoF = np.empty(nsrc + ndof)

        vDoF[:ndof] = self.node_voltages[isDoF]

        for nn in range(nsrc):
            matchArr = self._select_by_dof(nn + ndof)
            matches = np.nonzero(matchArr)[0]
            if matches.shape[0] > 0:
                sel = matches[0]
                vDoF[ndof + nn] = self.node_voltages[sel]

        return vDoF

    def _select_by_dof(self, dofNdx):
        nDoF = sum(self.node_role_table == 0)

        nCur = dofNdx - nDoF
        if nCur < 0:
            selector = np.zeros_like(self.node_role_table, dtype=bool)
        else:
            selector = np.logical_and(self.node_role_table == 2, self.node_role_values == nCur)

        return selector

    def calculate_analytical_voltage(self, rvec=None):
        """
        Analytical estimate of potential field.

        Calculates estimated potential from sum of piecewise functions

              Vsrc,         r<=rSrc
        v(r)={
              isrc/(4Pi*r)

        If rvec is none, calculates at every node of mesh

        Parameters
        ----------
        rvec : float[:], optional
            Distances from source to evaluate at. The default is None.

        Returns
        -------
        vAna, list of float[:]
            List (per source) of estimated potentials
        intAna, list of float
            Integral of analytical curve across specified range.

        """
        srcI = []
        source_locations = []
        srcRadii = []
        srcV = []

        for ii in nb.prange(len(self.current_sources)):
            source = self.current_sources[ii]
            I = source.value.get_value_at_time(self.current_time)
            rad = source.geometry.radius
            srcI.append(I)
            source_locations.append(source.geometry.center)
            srcRadii.append(rad)

            if rad > 0:
                V = I / (4 * np.pi * rad)
            srcV.append(V)

        for ii in nb.prange(len(self.voltage_sources)):
            source = self.voltage_sources[ii]
            V = source.value.get_value_at_time(self.current_time)
            srcV.append(V)
            source_locations.append(source.geometry.center)
            rad = source.radius
            srcRadii.append(rad)
            if rad > 0:
                I = V * 4 * np.pi * rad

            srcI.append(I)

        vAna = []
        intAna = []
        for ii in nb.prange(len(srcI)):
            if rvec is None:
                r = np.linalg.norm(self.mesh.node_coords - source_locations[ii], axis=1)
            else:
                r = rvec

            vEst, intEst = _analytic(srcRadii[ii], srcV[ii], srcI[ii], r)

            vAna.append(vEst)
            intAna.append(intEst)

        return vAna, intAna

    def estimate_volume_error(self, basic=False):
        """


        Parameters
        ----------
        basic : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        elVints : TYPE
            DESCRIPTION.
        elAnaInts : TYPE
            DESCRIPTION.
        elErrInts : TYPE
            DESCRIPTION.
        analyticInt : TYPE
            DESCRIPTION.

        """
        elVints = []
        elAnaInts = []
        elErrInts = []

        ana, anaInt = self.calculate_analytical_voltage()
        analyticInt = anaInt[0]

        netAna = np.sum(ana, axis=0)

        for el in self.mesh.elements:
            span = el.span
            if self.mesh.element_type == "Face":
                elInd = el.faces
            else:
                elInd = el.vertices
            globalInd = np.array([self.mesh.inverse_index_map[v] for v in elInd])
            elVals = self.node_voltages[globalInd]

            if self.mesh.element_type == "Face":
                vals = meshes.fem.interpolate_from_face(elVals, meshes.fem.HEX_VERTEX_COORDS)
                avals = np.abs(meshes.fem.interpolate_from_face(netAna[globalInd], meshes.fem.HEX_VERTEX_COORDS))
            else:
                vals = elVals
                avals = np.abs(netAna[globalInd])

            dvals = np.abs(vals - avals)

            if basic:
                vol = np.prod(span)
                intV = vol * np.mean(vals)
                intAna = vol * np.mean(avals)
                intErr = vol * np.mean(dvals)
            else:
                intV = meshes.FEM.integrate_from_verts(vals, span)
                intAna = meshes.FEM.integrate_from_verts(avals, span)
                intErr = meshes.FEM.integrate_from_verts(dvals, span)

            elErrInts.append(intErr)
            elVints.append(intV)
            elAnaInts.append(intAna)

        return elVints, elAnaInts, elErrInts, analyticInt

    def calculate_errors(self, universal_indices=None):
        """
        Estimate error in solution.

        Estimates error between simulated solution assuming point/spherical
        sources in uniform conductivity.

        The normalized error metric approximates the area between the
        analytical solution i/(4*pi*sigma*r) and a linear interpolation
        between the simulated nodal voltages, evaluated across the
        simulation domain

        Parameters
        ----------
        universal_indices : int[:], optional
            Alternative points at which to evaluate the analytical solution.
            The default is None.

        Returns
        -------
        errSummary : float
            Normalized, overall error metric.
        err : float[:]
            Absolute error estimate at each node (following global node ordering)
        vAna : float[:]
            Estimated potential at each node (following global ordering)
        sorter : int[:]
            Indices to sort globally-ordered array based on the corresponding node's distance from center
            e.g. erSorted=err[sorter]
        r : float[:]
            distance of each point from source
        """
        ind, v = self.get_universal_points()

        if universal_indices is not None:
            # r=np.linalg.norm(coords,axis=1)
            sel = np.isin(ind, universal_indices)
            v = v[sel]
            ind = ind[sel]

        if self.meshtype == "uniform":
            coord = self.mesh.node_coords
        else:
            coord = util.indices_to_coordinates(ind, origin=self.mesh.bbox[:3], span=self.mesh.span)
        r = np.linalg.norm(coord, axis=1)

        vEst, intAna = self.calculate_analytical_voltage(r)

        vAna = np.sum(np.array(vEst), axis=0)
        anaInt = abs(sum(intAna))

        sorter = np.argsort(r)
        rsort = r[sorter]

        err = v - vAna
        errSort = err[sorter]
        errSummary = np.trapz(abs(errSort), rsort) / anaInt

        return errSummary, err, vAna, sorter, r

    def __toDoF(self, globalIndex):
        role = self.node_role_table[globalIndex]
        roleVal = self.node_role_values[globalIndex]
        if role == 1:
            dofIndex = None
        else:
            dofIndex = roleVal
            if role == 2:
                dofIndex += self.nDoF

        return role, dofIndex

    def _get_node_types(self):
        """
        Get an integer per node indicating its role.

        Type indices:
            0: Unknown voltage
            1: Fixed voltage
            2: Fixed current, unknown voltage

        Returns
        -------
        None.

        """
        self.insert_sources_in_mesh()

        self.nDoF = sum(self.node_role_table == 0)
        trueDoF = np.nonzero(self.node_role_table == 0)[0]

        for n in nb.prange(len(trueDoF)):
            self.node_role_values[trueDoF[n]] = n

    def get_edge_matrix(self, dedup=True):
        """Return conductance matrix across all nodes in mesh.

        Parameters
        ----------
        dedup : bool, optional
            Sum parallel conductances. The default is True.

        Returns
        -------
        gAll : COO sparse matrix
            Conductance matrix, N x N for a mesh of N nodes.

        """
        nNodes = self.mesh.node_coords.shape[0]
        a = np.tile(self.conductances, 2)
        E = np.vstack((self.edges, np.fliplr(self.edges)))
        gAll = scipy.sparse.coo_matrix((a, (E[:, 0], E[:, 1])), shape=(nNodes, nNodes))
        if dedup:
            gAll.sum_duplicates()

        return gAll

    def count_node_connectivity(self, deduplicate=False):
        """
        Calculate how many conductances terminate in each node.

        A fully-connected hex node will have 24 edges prior to merging parallel
        conductances; less than this indicates the node is hanging (nonconforming).

        Parameters
        ----------
        deduplicate : boolean, optional
            Combine parallel conductances before counting, default False

        Returns
        -------
        nConn : int[:]
            Number of edges that terminate in each node.

        """
        if deduplicate:
            self._deduplicate_edges()
        _, nConn = np.unique(self.edges.ravel(), return_counts=True)

        if deduplicate:
            nConn[nConn > 6] = 6

        return nConn

    def regularize_mesh(self):
        """
        Not recommended: attempt to remove nonconforming nodes.

        Returns
        -------
        None.

        """
        nConn = self.count_node_connectivity()

        badNodes = np.argwhere(nConn < 24).squeeze()
        keepEdge = np.ones(self.edges.shape[0], dtype=bool)
        boundary_nodes = self.mesh.get_boundary_nodes()

        hangingNodes = np.setdiff1d(badNodes, boundary_nodes)

        new_edges = []
        new_conductances = []

        for ii in nb.prange(hangingNodes.shape[0]):
            node = hangingNodes[ii]

            # get edges connected to hanging node
            isSharedE = np.any(self.edges == node, axis=1, keepdims=True)
            isOther = self.edges != node
            neighbors = self.edges[isSharedE & isOther]
            # get edges connected adjacent to hanging node
            matchesNeighbor = np.isin(self.edges, neighbors)
            isLongEdge = np.all(matchesNeighbor, axis=1)

            # get long edges (forms triangle with edges to hanging node)
            longEdges = self.edges[isLongEdge]
            gLong = self.conductances[isLongEdge]

            # TODO: generalize split
            for eg, g in zip(longEdges, gLong):
                for n in eg:
                    new_edges.append([n, node])
                    new_conductances.append(0.5 * g)

            keepEdge[isLongEdge] = False

        if len(new_edges) > 0:
            revisedEdges = np.vstack((self.edges[keepEdge], np.array(new_edges)))
            revisedConds = np.concatenate((self.conductances[keepEdge], np.array(new_conductances)))

            self.conductances = revisedConds
            self.edges = revisedEdges

    def _deduplicate_edges(self):
        e = self.edges
        g = self.conductances
        nnodes = self.mesh.node_coords.shape[0]

        gdup = np.concatenate((g, g))
        edup = np.vstack((e, np.fliplr(e)))

        a, b = np.hsplit(edup, 2)

        gmat = scipy.sparse.coo_matrix((gdup, (edup[:, 0], edup[:, 1])), shape=(nnodes, nnodes))
        gmat.sum_duplicates()

        tmp = scipy.sparse.tril(gmat, -1)

        gComp = tmp.data
        eComp = np.array(tmp.nonzero()).transpose()

        self.edges = eComp
        self.conductances = gComp

    def _is_matching_edge(self, edges, toMatch):
        nodeMatches = np.isin(edges, toMatch)
        matchingEdge = np.all(nodeMatches, axis=1)
        return matchingEdge

    def _construct_system(self):
        """
        Construct system of equations GV=b.

        Rows represent each node without a voltage or current
        constraint, followed by an additional row per current
        source.

        Returns
        -------
        G : COO sparse matrix
            Conductances between degrees of freedom.
        b : float[:]
            Right-hand side of system, representing injected current
            and contributions of fixed-voltage nodes.

        """
        # global mat is NxN
        # for Ns current sources, Nf fixed nodes, and Nx floating nodes,
        # N - nf = Ns + Nx =Nd
        # system is Nd x Nd

        # isSrc=self.node_role_table==2
        isFix = self.node_role_table == 1

        # N=self.mesh.node_coords.shape[0]
        # Ns=len(self.current_sources)
        # Nx=util.fastcount(self.node_role_table==0)
        # Nd=Nx+Ns
        # Nf=util.fastcount(isFix)

        self.start_timing("Filter conductances")
        # #renumber nodes in order of dof, current source, fixed v
        # dofNumbering=self.node_role_values.copy()
        # dofNumbering[isSrc]=Nx+dofNumbering[isSrc]
        # dofNumbering[isFix]=Nd+np.arange(Nf)

        dofNumbering, Nset = self._get_ordering("dof")
        Nx, Nf, Ns, Nd = Nset
        N_ext = Nd + Nf

        edges = dofNumbering[self.edges]

        # filter bad vals (both same DoF)
        isValid = edges[:, 0] != edges[:, 1]
        evalid = edges[isValid]
        cvalid = self.conductances[isValid]

        # duplicate for symmetric matrix
        Edup = np.vstack((evalid, np.fliplr(evalid)))
        cdup = np.tile(cvalid, 2)

        # get only DOF rows/col
        isRowDoF = Edup[:, 0] < Nd

        # Fill matrix with initial degrees of freedom
        E = Edup[isRowDoF]
        c = cdup[isRowDoF]

        self.log_time()
        self.start_timing("Assemble system")
        G = scipy.sparse.coo_matrix((-c, (E[:, 0], E[:, 1])), shape=(Nd, N_ext))

        gR = G.tocsr()

        v = np.zeros(N_ext)
        v[Nd:] = self.node_voltages[isFix]

        b = -np.array(gR.dot(v)).squeeze()

        for ii in range(Ns):
            # b[ii+Nx]+=self.current_sources[ii].value
            b[ii + Nx] = self._node_current_sources[ii]

        # idx=self.node

        diags = -np.array(gR.sum(1)).squeeze()

        G.setdiag(diags)
        G.resize(Nd, Nd)

        self.log_time()

        self.RHS = b
        self.gMat = G
        return G, b

    def get_coordinates_in_order(self, order_type="mesh", mask_array=None):
        """
        Get mesh node coordinates according to specified ordering scheme.

        Parameters
        ----------
        order_type : str, optional
            'mesh'       - Sorted by universal index (default)
            'dof'        - Sorted by unknowns in system of equations
                           (floating nodes by universal index, then
                           current-source nodes in order of source)
            'electrical' - Like mesh, but current-source nodes moved to
                            end and replaced by source center
        mask_array : bool[:] or None, optional
            Use only mesh nodes where index is True, or use all if None

        Returns
        -------
        float[:,3]
            Coordinates in specified ordering scheme, possibly masked.
        """
        if order_type == "mesh":
            reorderCoords = self.mesh.node_coords

        if order_type == "dof":
            ordering, _ = self._get_ordering("dof")
            dofCoords = self.mesh.node_coords[self.node_role_table == 0]
            source_coords = np.array([s.geometry.center for s in self.current_sources])
            reorderCoords = np.vstack((dofCoords, source_coords))

        if order_type == "electrical":
            ordering, (Nx, Nf, Ns, Nd) = self._get_ordering("electrical")
            universal_voltagesals, valid = np.unique(ordering, return_index=True)
            if mask_array is None:
                coords = self.mesh.node_coords
            else:
                coords = np.ma.array(self.mesh.node_coords, mask=~mask_array)

            reorderCoords = np.empty((Nx + Nf + Ns, 3))

            nonSrc = self.node_role_table < 2
            reorderCoords[: Nx + Nf] = coords[nonSrc]
            for ii in nb.prange(len(self.current_sources)):
                reorderCoords[Nx + Nf + ii] = self.current_sources[ii].coords

        return reorderCoords

    def get_edges_in_order(self, order_type="mesh", mask_array=None):
        """
        Get mesh edges with nodes ordered by specified scheme.

        Parameters
        ----------
        order_type : str, optional
            'mesh'       - Sorted by universal index (default)
            'dof'        - Sorted by unknowns in system of equations
                           (floating nodes by universal index, then
                           current-source nodes in order of source)
            'electrical' - Like mesh, but current-source nodes moved to
                            end and replaced by source center
        mask_array : bool[:] or None, optional
            Use only mesh nodes where index is True, or use all if None

        Returns
        -------
        int[:,2]
            Edges with indices from specified ordering
        """
        if order_type == "mesh":
            edges = self.edges
        if order_type == "electrical":
            ordering, (Nx, Nf, Ns, Nd) = self._get_ordering(order_type)
            isSrc = ordering < 0
            ordering[isSrc] = Nx + Nf - 1 - ordering[isSrc]
            if mask_array is None:
                oldEdges = self.edges
            else:
                okEdge = np.all(mask_array[oldEdges], axis=1)
                oldEdges = self.edges[okEdge]

            # edges=oldEdges.copy()
            isSrc = self.node_role_table == 2
            order = np.empty(isSrc.shape[0], dtype=np.int64)
            order[~isSrc] = np.arange(Nx + Nf)
            order[isSrc] = self.node_role_values[isSrc] + Nx + Nf
            edges = order[oldEdges]

        return edges

    def get_mesh_geometry(self):
        """
        Get node and edge topology from mesh.

        Returns
        -------
        coords : float[:,3]
            Cartesian coordinates of mesh nodes
        edges : uint64[:,2]
            Indices of endpoints for each edge in mesh
        """
        verts = []
        raw_edgess = []
        for el in self.mesh.elements:
            vert = el.vertices
            edge = vert[ADMITTANCE_EDGES]
            verts.extend(vert.tolist())
            raw_edgess.extend(edge)

        inds = np.unique(np.array(verts, dtype=np.uint64))
        coords = util.indices_to_coordinates(inds, self.mesh.bbox[:3], self.mesh.span)

        edges = util.renumber_indices(np.array(raw_edgess, dtype=np.uint64), inds)

        return coords, edges

    def interpolate_at_points(self, coords, elements=None, data=None):
        """
        Interpolate values at specified coordinates.

        Parameters
        ----------
        coords : float[:,:]
            Coordinates to interpolate ate.
        elements : element list, optional
            Elements to search for the proper interpolation. The default is None, which checks all elements.
        data : float[:], optional
            Nodal values used for interpolation. The default is None, which uses node voltages.

        Returns
        -------
        vals : float[:]
            Interpolated values at the specfied points

        """
        vals = np.zeros(coords.shape[0])
        unknown = np.ones_like(vals, dtype=bool)

        if elements is None:
            elements = self.mesh.elements

        if data is None:
            data = self.node_voltages

        for el in elements:
            upper = np.greater_equal(coords, el.origin)
            lower = np.less_equal(coords, el.origin + el.span)
            inside = np.all(np.logical_and(upper, lower), axis=1)

            newpt = np.logical_and(inside, unknown)
            intCoords = coords[newpt]

            if intCoords.shape[0] > 0:
                if self.mesh.element_type == "Face":
                    inds = el.faces
                else:
                    inds = el.vertices

                simInds = np.array([self.mesh.inverse_index_map[n] for n in inds])
                elValues = data[simInds]

                interpVals = el.interpolate_within(intCoords, elValues)

                vals[newpt] = interpVals
                unknown = np.logical_or(unknown, inside)

                if not np.any(unknown):
                    return vals

                # coordsLeft[inside,:]=np.ma.masked

        return vals

    def _get_ordering(self, order_type):
        """
        Get integer tag for each node according to the designated numbering scheme.

        Parameters
        ----------
        order_type : string
            Whether to order by corresponding degree of freedom 'dof'
            or electrical .

        Returns
        -------
        int[:]
            Tag for each node.

        """
        isSrc = self.node_role_table == 2
        isFix = self.node_role_table == 1

        # N=self.mesh.node_coords.shape[0]
        # Ns=util.fastcount(np.nonzero(self.node_role_table==2))
        # Ns=len(self.current_sources)
        Ns = len(self._node_current_sources)
        Nx = util.fastcount(self.node_role_table == 0)
        Nd = Nx + Ns
        Nf = util.fastcount(isFix)

        # if order_type=='electrical':
        if order_type == "dof":
            # renumber nodes in order of dof, current source, fixed v
            numbering = self.node_role_values.copy()
            numbering[isSrc] = Nx + numbering[isSrc]
            numbering[isFix] = Nd + np.arange(Nf)

        if order_type == "electrical":
            numbering = self.node_role_values.copy()
            numbering[isFix] = Nx + np.arange(Nf)
            numbering[isSrc] = -1 - numbering[isSrc]

        return numbering, (Nx, Nf, Ns, Nd)

    # TODO: deprecate and remove
    def get_elements_in_plane(self, axis=2, point=0.0):
        """
        Get all elements that intersect a plane orthogonal to the axes.

        .. deprecated
            Use PyVista slicing routines instead for greater
            robustness and flexibility.

        Parameters
        ----------
        axis : int, optional
            DESCRIPTION. The default is 2.
        point : float, optional
            DESCRIPTION. The default is 0..

        Returns
        -------
        elements : elements
            DESCRIPTION.
        coords : float[:,2]
        edgePts : float[:,2,2]

        """
        otherAx = np.array([n != axis for n in range(3)])
        # arrays=[]
        # Gmax=self.mesh.max_depth+1

        # closestPlane=int((point-self.mesh.bbox[axis])/self.mesh.span[axis])
        # originIdx=util.position_to_index(np.array([0,0,closestPlane]),
        #                          2**Gmax+1)
        origin = self.mesh.bbox.copy()[:3]
        origin[axis] = point

        elements = self.mesh.get_intersecting_elements(axis, coordinate=point)

        coords = []
        edgePts = []
        for ii, el in enumerate(elements):
            ori = el.origin[otherAx]
            ext = el.span[otherAx] + ori
            q = [ori, ext]

            elCoords = np.array([[q[b][0], q[a][1]] for a in range(2) for b in range(2)])

            edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
            edgePts.extend(elCoords[edges])
            coords.extend(elCoords)

        return elements, np.array(coords), np.array(edgePts)

    def _getUniformPlane(self, axis, point, data):
        inplane = self.mesh.node_coords[:, axis] == point

        otherAxes = np.array([n != axis for n in range(3)])

        planeCoords = self.mesh.node_coords[:, otherAxes][inplane]

        vals = data[inplane]

        nx = int(np.sqrt(vals.shape[0]))

        return [vals.reshape((nx, nx))], planeCoords

    # TODO: deprecate and remove
    def getValuesInPlane(self, axis=2, point=0.0, data=None):
        """
        Extract values in a plane.

        .. deprecated
            Use PyVista slicing routines instead for greater robustness.

        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 2.
        point : TYPE, optional
            DESCRIPTION. The default is 0..
        data : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if data is None:
            data = self.node_voltages

        if self.meshtype == "uniform":
            return self._getUniformPlane(axis=axis, point=point, data=data)

        elements, coords, _ = self.get_elements_in_plane(axis, point)

        depths = np.array([el.depth for el in elements])

        dcats = np.unique(depths)
        nDcats = dcats.shape[0]

        element_lists = (max(dcats) + 1) * [[]]

        for el, d in zip(elements, depths):
            element_lists[d].append(el)

        whichInds = np.array([[0, 2, 4, 6], [0, 1, 4, 5], [0, 1, 2, 3]])
        selInd = whichInds[axis]

        notAx = [ax != axis for ax in range(3)]

        mask_arrays = []
        for ii in nb.prange(nDcats):
            els = element_lists[ii]
            if len(els) == 0:
                continue

            nX = 2 ** (dcats[ii]) + 1
            # arr0=np.nan*np.empty((nX,nX))
            arr = np.ma.masked_all((nX, nX))

            pts = []
            vals = []
            xx = []
            yy = []

            for ee in nb.prange(len(elements)):
                el = elements[ee]
                if el.depth != dcats[ii]:
                    continue

                if self.mesh.element_type == "Face":
                    element_universal_indices = el.faces
                else:
                    element_universal_indices = el.vertices

                nodes = [self.mesh.inverse_index_map[n] for n in element_universal_indices]
                if len(nodes) == 0:
                    print("empty element:")
                    print(el.index)
                    print("verts: " + str(el.vertices))
                    print("faces: " + str(el.faces))
                    continue
                interp = el.get_planar_values(data[nodes], axis=axis, coord=point)

                # xy0=util.octant_list_to_xyz(np.array(el.index))[notAx]
                xy0 = util.reverse_octant_list_to_xyz(np.array(el.index))[notAx]

                xys = np.array([xy0.astype(np.int_) + np.array([x, y]) for y in [0, 1] for x in [0, 1]])
                pts.extend(nodes)
                vals.extend(interp)
                # xy.extend(xys)

                x, y = np.hsplit(xys, 2)
                # arr0[x,y]=interp
                xx.extend(x)
                yy.extend(y)

                if np.any(xy0 >= nX):
                    print()

                for i, j, v in zip(y, x, interp):
                    arr[i, j] = v

            # arr=scipy.sparse.coo_matrix((vals,(xx,yy)))
            # marr=np.ma.masked

            # mask_arrays.append(np.ma.masked_invalid(arr0))
            mask_arrays.append(arr)

        return mask_arrays, coords

    def get_universal_points(self, elements=None):
        """
        Get index and voltage at every universal point in the elements.

        Parameters
        ----------
        elements : list of :py:class:`xcell.mesh.Element`, optional
            Elements to extract points from, or all elements in mesh if
            None (default)

        Returns
        -------
        universal_indices : uint64[:]
            Universal indices of the elements
        universal_voltages : float[:]
            Voltage at each universal point, interpolated if necessary
        """
        if self.meshtype == "uniform":
            universal_indices = self.mesh.index_map
            universal_voltages = self.node_voltages
        else:
            if elements is None:
                elements = self.mesh.elements

            universalIndices = []
            universalVals = []
            for ii in nb.prange(len(elements)):
                el = self.mesh.elements[ii]

                if self.mesh.element_type == "Face":
                    element_universal_indices = el.faces
                else:
                    element_universal_indices = el.vertices

                vals = np.array(
                    [self.node_voltages[self.mesh.inverse_index_map[nn]] for nn in element_universal_indices]
                )
                uVal, uInd = el.get_universal_vals(vals)

                universalIndices.extend(uInd.tolist())
                universalVals.extend(uVal.tolist())

            # explicit use of uint required; casts to float otherwise
            indArr = np.array(universalIndices, dtype=np.uint64)

            universal_indices, invmap = np.unique(indArr, return_index=True)

            universal_voltages = np.array(universalVals)[invmap]

        return universal_indices, universal_voltages

    def getCurrentsInPlane(self, axis=2, point=0.0):
        """
        Get currents through edges within plane.

        .. deprecated
            PyVista routines recommended instead

        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 2.
        point : TYPE, optional
            DESCRIPTION. The default is 0..

        Returns
        -------
        currents : TYPE
            DESCRIPTION.
        currentPts : TYPE
            DESCRIPTION.
        mesh : float[:,:,2]
            XY coordinates of mesh nodes in plane.

        """
        els, coords, mesh = self.get_elements_in_plane(axis, point)

        if self.mesh.element_type == "Face":
            inds = np.unique([el.faces for el in els])
        else:
            inds = np.unique([el.vertices for el in els])

        dic = util.get_py_dict(self.mesh.index_map)
        gInds = [dic[i] for i in inds]

        inPlane = np.all(np.isin(self.edges, gInds), axis=1)

        dv = np.diff(self.node_voltages[self.edges[inPlane]], axis=1)

        i = self.conductances[inPlane] * dv.squeeze()

        ineg = i < 0

        currents = i.copy()
        currents[ineg] = -currents[ineg]

        otherAx = [n != axis for n in range(3)]
        pcoord = self.mesh.node_coords[:, otherAx]

        corEdge = self.edges[inPlane]
        corEdge[ineg, :] = np.fliplr(corEdge[ineg, :])

        currentPts = pcoord[corEdge]

        return currents, currentPts, mesh


class Study:
    """IO manager for multiple related simulations"""

    def __init__(self, study_path, bounding_box):
        """
        Create manager.

        Parameters
        ----------
        study_path : pathlike
            Path to folder containing study data
        bounding_box : float[6]
            Bounding box of simulation domain
        """
        if not os.path.exists(study_path):
            os.makedirs(study_path)
        self.study_path = study_path

        self.nSims = -1
        self.current_simulation = None
        self.bbox = bounding_box
        self.span = bounding_box[3:] - bounding_box[:3]
        self.center = bounding_box[:3] + self.span / 2

        # self.iSourceCoords = []
        # self.iSourceVals = []
        # self.vSourceCoords = []
        # self.vSourceVals = []

    def new_simulation(self, simName=None, keepMesh=False):
        """
        Create new simulation in study.

        Parameters
        ----------
        simName : string, optional
            Name of new simulation, or auto-increment 'sim%d' if None
            (default)
        keepMesh : bool, optional
            Retain original mesh instead of regenerating, by default False

        Returns
        -------
        :py:class:`.Simulation`
            New simulation
        """
        self.nSims += 1

        if simName is None:
            simName = "sim%d" % self.nSims

        sim = Simulation(simName, bbox=self.bbox)
        if keepMesh:
            sim.mesh = self.current_simulation.mesh
        # sim.mesh.extents=self.span

        self.current_simulation = sim

        return sim

    def save_mesh(self, simulation=None):
        """
        Save mesh to file

        Parameters
        ----------
        simulation : `.Simulation`, optional
            Simulation to save, or self.current_sim if None (default)
        """
        if simulation is None:
            simulation = self.current_simulation

        mesh = simulation.mesh
        num = str(simulation.meshnum)

        fname = os.path.join(self.study_path, "mesh" + num + ".p")
        pickle.dump(mesh, open(fname, "wb"))

    def load_mesh(self, meshnum):
        """
        Reload mesh from file

        Parameters
        ----------
        meshnum : int
            Index of simulation in study

        Returns
        -------
        `xcell.meshes.Mesh`
            Reloaded mesh
        """
        fstem = "mesh" + str(meshnum) + ".p"
        fname = os.path.join(self.study_path, fstem)

        mesh = pickle.load(open(fname, "rb"))

        self.current_simulation.mesh = mesh

        return mesh

    def log_current_simulation(self, extraCols=None, extraVals=None):
        """
        Log current simulation stats to csv file.

        Parameters
        ----------
        extraCols : string[:], optional
            Additional column labels. The default is None.
        extraVals : [:], optional
            Additional column data. The default is None.

        Returns
        -------
        None.

        """
        fname = os.path.join(self.study_path, "log.csv")
        self.current_simulation.log_table_entry(fname, extraCols=extraCols, extraVals=extraVals)

    # TODO: see if removable in saving logic
    def save_simulation(self, simulation, baseName=None, addedTags=""):
        """
        Save

        Parameters
        ----------
        simulation : _type_
            _description_
        baseName : _type_, optional
            _description_, by default None
        addedTags : str, optional
            _description_, by default ''
        """
        data = {}

        meshpath = os.path.join(self.study_path, str(simulation.meshnum) + ".p")

        if ~os.path.exists(meshpath):
            self.save_mesh(simulation)
        else:
            simulation.mesh = None

        if baseName is None:
            baseName = simulation.name

        fname = os.path.join(self.study_path, baseName + addedTags + ".p")
        pickle.dump(simulation, open(fname, "wb"))

    def load_simulation(self, simName):
        fname = self._get_file_path(simName)
        data = pickle.load(open(fname, "rb"))
        if data.mesh is None:
            meshpath = self._get_file_path("mesh" + str(data.meshnum))
            mesh = pickle.load(open(meshpath, "rb"))
            data.mesh = mesh

        return data

    def save(self, obj, fname, ext=".p"):
        fpath = self.__makepath(fname, ext)
        if "read" in dir(obj):
            obj.read(fpath)
        else:
            pickle.dump(obj, open(fpath, "wb"))

    def load(self, fname, ext=".p"):
        fpath = self._get_file_path(fname, ext)
        obj = pickle.load(open(fpath, "rb"))
        return obj

    def _get_file_path(self, name, extension=".p"):
        filepath = os.path.join(self.study_path, name + extension)
        return filepath

    def save_plot(self, fig, file_name, ext=None):
        """
        Save matplotlib plot in study folder

        Parameters
        ----------
        fig : :py:class:`matplotlib.figure.Figure`
            Figure to save
        file_name : string
            _description_
        ext : string, optional
            File format to use, or save as .png, .svg, and .eps if None
            (default)
        """
        if ext is None:
            fnames = [self.__makepath(file_name, x) for x in [".png", ".svg", ".eps"]]

        else:
            fnames = [self.__makepath(file_name, ext)]

        for f in fnames:
            fig.savefig(f, transparent=True, dpi=300)

    def __makepath(self, file_name, ext):
        fpath = os.path.join(self.study_path, file_name + ext)

        basepath, _ = os.path.split(fpath)

        if not os.path.exists(basepath):
            os.makedirs(basepath)
        return fpath

    def save_animation(self, animator, filename):
        """
        Save matplotlib-based animation (not exported movie!) for reuse.

        Parameters
        ----------
        animator : :py:class:`xcell.visualizers.FigureAnimator`
            Animator to save.
        filename : string
            Name of pickled file.

        Returns
        -------
        None.

        """
        fname = self.__makepath(filename, ".adata")
        pickle.dump(animator, open(fname, "wb"))

    def save_pv_image(self, plotter, filename, **kwargs):
        """
        Save PyVista plot to image.

        Parameters
        ----------
        plotter : PyVista Plotter
            Active plotter.
        filename : str
            File name, with or without extension.
            Saves as .pdf if not specified.
        ** kwargs :
            Options for plotter.show()

        Returns
        -------
        None.

        """
        f, ext = os.path.splitext(filename)
        if ext == "":
            ext = ".pdf"
        fname = self.__makepath(f, ext)

        if ext == ".png":
            plotter.screenshot(fname, **kwargs)
        else:
            plotter.save_graphic(fname, **kwargs)

    def make_pv_movie(self, plotter, filename, **kwargs):
        """
        Open movie file for PyVista animation.

        Parameters
        ----------
        plotter : PyVista Plotter
            Active plotter.
        filename : str
            File name, with or without extension.
            Saves as .mp4 if not specified.
        ** kwargs :
            Options for plotter.open_movie()

        Returns
        -------
        None.

        """
        f, ext = os.path.splitext(filename)
        if ext == "":
            ext = ".mp4"
        fname = self.__makepath(f, ext)

        plotter.open_movie(fname, **kwargs)

    # TODO: merge with visualizers.import_logfile?
    def load_logfile(self):
        """
        Returns Pandas dataframe of logged runs

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        cats : TYPE
            DESCRIPTION.

        """
        logfile = os.path.join(self.study_path, "log.csv")
        df, cats = visualizers.import_logfile(logfile)
        return df, cats

    # TODO: seems to be dup functionality....
    def get_saved_simulations(self, filter_categories=None, filter_values=None, group_category=None):
        """
        Get all simulations in study, grouped and filtered as specified.

        Parameters
        ----------
        filter_categories : list of strings, optional
            Categories in dataset to filter by, or use all if None (default)
        filter_values : list of any, optional
            Values to match for each filter category. The default is None,
            which matches all
        group_category : string, optional
            Key to group simulations by, or no grouping if None (default)

        Returns
        -------
        fnames : list of lists of strings
            Simulation filenames matching each filter and group value.
        categories : list of strings
            Unique keys in group_category, or None if ungrouped

        """

        logfile = os.path.join(self.study_path, "log.csv")
        df, cats = visualizers.import_logfile(logfile)

        selector = np.ones(len(df), dtype=bool)
        if filter_categories is not None:
            for cat, val in zip(filter_categories, filter_values):
                selector &= df[cat] == val

        if group_category is not None:
            sortcats = df[group_category]
            sortvals = sortcats.unique()

            fnames = []
            categories = []
            for val in sortvals:
                sorter = df[group_category] == val
                fnames.append(df["File name"][sorter & selector])

                categories.append(val)

        else:
            fnames = [df["File name"][selector]]
            categories = [None]

        return fnames, categories


# def make_bounded_linear_metric(min_l0, max_l0, domain_x, origin=np.zeros(3)):
#     """
#     FIXME: update to new signature

#     Parameters
#     ----------
#     min_l0 : float
#         Smallest l0 permitted
#     max_l0 : float
#         Largest l0 permitted
#     domain_x : float
#         _description_
#     origin : float[3], optional
#         _description_, by default np.zeros(3)

#     Returns
#     -------
#     metric
#         Numba-compiled metric function
#     """
#     @nb.njit()
#     def metric(coord, a=min_l0, k=max_l0/domain_x):
#         r = np.linalg.norm(coord-origin)
#         val = a+r*k
#         return val

#     return metric


# def makeExplicitLinearMetric(maxdepth, meshdensity, origin=np.zeros(3)):
#     """
#     FIXME: update to new signature

#     Parameters
#     ----------
#     maxdepth : _type_
#         _description_
#     meshdensity : _type_
#         _description_
#     origin : _type_, optional
#         _description_, by default np.zeros(3)

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     param = 2**(-maxdepth*meshdensity)  # -1)*3**(0.5)
#     # @nb.njit()

#     def metric(coord):
#         r = np.linalg.norm(coord-origin)
#         val = r*param
#         return val

#     return metric


def _analytic(rad, V, I, r):
    inside = r < rad
    voltage = np.empty_like(r)
    voltage[inside] = V
    voltage[~inside] = I / (4 * np.pi * r[~inside])

    integral = V * rad * (1 + np.log(max(r) / rad))
    return voltage, integral


# def makeScaledMetrics(maxdepth, density=0.2):
#     param = 2**(-maxdepth*density)
# # FIXME: match signature
#     @nb.njit()
#     def metric(elBBox, reference_coords, coefs):
#         nPts = reference_coords.shape[0]
#         coord = (elBBox[:3]+elBBox[3:])/2
#         l0s = np.empty(nPts)

#         for ii in nb.prange(nPts):
#             l0s[ii] = param*coefs[ii]*np.linalg.norm(reference_coords[ii]-coord)

#         return l0s

#     return metric


@nb.njit()
def general_metric(element_bbox, reference_coords, reference_coefficents):
    """
    Standard linear size metric for element.

    Parameters
    ----------
    element_bbox : float[6]
        Element bounding box in xcell order
    reference_coords : float[:,3]
        Cartesian coordinates of reference points
    reference_coefficents : float[:]
        Coefficents of distance for each reference point

    Returns
    -------
    l0s : float[:]
        Target size according to each reference/coefficent pair
    """

    nPts = reference_coords.shape[0]
    coord = (element_bbox[:3] + element_bbox[3:]) / 2
    l0s = np.empty(nPts)

    for ii in nb.prange(nPts):
        l0s[ii] = reference_coefficents[ii] * np.linalg.norm(reference_coords[ii] - coord)

    return l0s


def get_standard_mesh_params(sources, mesh_depth, density=0.2):
    """
    Generate default inputs to general metric for given parameters.

    Parameters
    ----------
    sources : list of xcell sources
        Sources providing reference points.
    mesh_depth : int
        Maximum splitting depth for mesh.
    density : float, optional
        Density of mesh (0.0 = maximally sparse, 1.0 = maximally dense).
        The default is 0.2.

    Returns
    -------
    source_coords : float[:,3]
        Coords of reference points.
    max_depths : int[:]
        Maximum depth allowed per point.
    coefs : float[:]
        Coefficents per point.

    """
    nSrc = len(sources)

    coefs = np.ones(nSrc) * 2 ** (-density * mesh_depth)
    max_depths = np.ones(nSrc, dtype=int) * mesh_depth

    srcPts = []
    for src in sources:
        if "geometry" in dir(src):
            srcPts.append(src.geometry.center)
        else:
            srcPts.append(src.coords)

    source_coords = np.array(srcPts, ndmin=2)

    return source_coords, max_depths, coefs
