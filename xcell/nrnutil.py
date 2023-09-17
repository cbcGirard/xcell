#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for interfacing with NEURON."""


from neuron import h
import neuron.units as nUnit

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from . import colors

from .xCell import Simulation, get_standard_mesh_params, general_metric
from . import util


def set_v_ext(vext, vstim, tstim):
    """
    Set fixed extracellular potentials at all neural compartments.

    Parameters
    ----------
    vext : float[:,:]
        Vector of external voltages.
    vstim : `neuron.Vector`
        Vector of governing stimulus values.
    tstim : `neuron.Vector`
        Vector of timestamps.

    Returns
    -------
    `neuron.Vector`
        Tuple of extracellular and intracellular potentials at all
        compartments.
    """
    vvecs = []
    vmems = []
    for sec, V in zip(h.allsec(), vext):
        for seg in sec.allseg():
            vseg = V * nUnit.V

            vvecs.append(h.Vector(vseg * vstim.as_numpy()))
            vvecs[-1].play(seg.extracellular._ref_e, tstim, False)
            vmems.append(h.Vector().record(seg._ref_v))

    return vvecs, vmems


def return_segment_coordinates(section, in_microns=False):
    """
    Get geometry info at segment centers.

    Adapted from https://www.neuron.yale.edu/phpBB/viewtopic.php?p=19176#p19176

    Modified to give segment radius as well

    Parameters
    ----------
    section : NEURON section
        The section to return info about.
    in_microns : bool, default False
        Whether to return values in microns (NEURON default)
        or in meters (xcell default)

    Returns
    -------
    xCoord : float
        x coordinate.
    yCoord : float
        y coordinate.
    zCoord : float
        z coordinate.
    rads : float
        radius of segment

    """
    # Get section 3d coordinates and put in numpy array
    n3d = section.n3d()
    x3d = np.empty(n3d)
    y3d = np.empty(n3d)
    z3d = np.empty(n3d)
    rad = np.empty(n3d)
    L = np.empty(n3d)
    for i in range(n3d):
        x3d[i] = section.x3d(i)
        y3d[i] = section.y3d(i)
        z3d[i] = section.z3d(i)
        rad[i] = section.diam3d(i) / 2

    # Compute length of each 3d segment
    for i in range(n3d):
        if i == 0:
            L[i] = 0
        else:
            L[i] = np.sqrt((x3d[i] - x3d[i - 1]) ** 2 + (y3d[i] - y3d[i - 1]) ** 2 + (z3d[i] - z3d[i - 1]) ** 2)

    # Get cumulative length of 3d segments
    cumLength = np.cumsum(L)

    N = section.nseg

    if N == 1:
        # special case of single segment, e.g. a soma
        xCoord = np.array(x3d[1])
        yCoord = np.array(y3d[1])
        zCoord = np.array(z3d[1])
        rads = np.array(rad[1])

    else:
        # Now upsample coordinates to segment locations
        xCoord = np.empty(N)
        yCoord = np.empty(N)
        zCoord = np.empty(N)
        rads = np.empty(N)
        dx = section.L / (N - 1)
        for n in range(N):
            if n == N - 1:
                xCoord[n] = x3d[-1]
                yCoord[n] = y3d[-1]
                zCoord[n] = z3d[-1]
                rads[n] = rad[-1]
            else:
                # which idx of 3d segments are we starting at
                cIdxStart = np.where(n * dx >= cumLength)[0][-1]
                # how far along that segment is this upsampled coordinate
                cDistFrom3dStart = n * dx - cumLength[cIdxStart]
                # what's the fractional distance along this 3d segment
                cFraction3dLength = cDistFrom3dStart / L[cIdxStart + 1]
                # compute x and y positions
                xCoord[n] = x3d[cIdxStart] + cFraction3dLength * (x3d[cIdxStart + 1] - x3d[cIdxStart])
                yCoord[n] = y3d[cIdxStart] + cFraction3dLength * (y3d[cIdxStart + 1] - y3d[cIdxStart])
                zCoord[n] = z3d[cIdxStart] + cFraction3dLength * (z3d[cIdxStart + 1] - z3d[cIdxStart])
                rads[n] = rad[cIdxStart] + cFraction3dLength * (rad[cIdxStart + 1] - rad[cIdxStart])

    if in_microns:
        scale = 1.0
    else:
        scale = 1e-6

    return xCoord * scale, yCoord * scale, zCoord * scale, rads * scale


def get_neuron_geometry():
    """
    Get geometric info of all compartments.

    Returns
    -------
    coords : float[:,3]
        Cartesian coordinates of compartment centers.
    rads : float[:]
        Radius of each compatment.
    is_sphere : bool[:]
        Whether compartment is assumed to represent a sphere.

    """
    coords = []
    rads = []
    is_sphere = []
    for sec in h.allsec():
        N = sec.n3d() - 1
        x, y, z, r = return_segment_coordinates(sec)

        coord = np.vstack((x, y, z)).transpose()
        coords.extend(coord)
        if coord.shape[0] == 1:
            rads.append(r.tolist())
        else:
            rads.extend(r.tolist())
        if N > 0:
            for ii, seg in enumerate(sec.allseg()):
                if ii == 0 or ii == N:
                    continue
                else:
                    sph = sec.hname().split(".")[-1] == "soma"

                    is_sphere.append(sph)
    return coords, rads, is_sphere


def get_membrane_currents():
    """
    Record total membrane current of every compartment.

    Returns
    -------
    ivecs : list of Vector
        List of vectors of membrane current.

    """
    ivecs = []
    for sec in h.allsec():
        N = sec.n3d() - 1
        if N > 0:
            for ii, seg in enumerate(sec.allseg()):
                if ii == 0 or ii == N:
                    continue
                else:
                    ivec = h.Vector().record(seg._ref_i_membrane_)

                    ivecs.append(ivec)

    return ivecs


def show_cell_geo(axis, polys=None, show_nodes=False):
    """
    Add cell geometry to designated plot.

    Parameters
    ----------
    axis : matplotlib axis
        Axis to plot on.
    polys : List of M x 2 arrays, optional
        Polygons to plot. The default is None,
        which queries NEURON for all compartments.

    Returns
    -------
    polys : List of M x 2 arrays
        Matplotlib representation of polygons

    """
    shade = colors.FAINT
    if polys is None:
        polys = get_cell_image()

    polycol = PolyCollection(polys, color=shade)
    axis.add_collection(polycol)
    if show_nodes:
        coords = np.array([return_segment_coordinates(sec) for sec in h.allsec()])
        axis.scatter(coords[:, 0], coords[:, 1], color="C0", marker="*")

    return polys


def get_cell_image():
    """
    Get plottable representation of cell geometry.

    Returns
    -------
    polys : List of M x 2 arrays
        List of vertices for creating a
        `matplotlib.collections.PolyCollection`.

    """
    tht = np.linspace(0, 2 * np.pi)

    polys = []
    for sec in h.allsec():
        x, y, z, r = return_segment_coordinates(sec)
        coords = np.vstack((x, y, z)).transpose()

        if sec.hname().split(".")[-1] == "soma":
            sx = x + r * np.cos(tht)
            sy = y + r * np.sin(tht)

            # axis.fill(sx, sy, color=shade)
            pts = np.vstack((sx, sy))
            polys.append(pts.transpose())

        else:
            # lx = x.shape
            lx = coords.shape[0]
            if lx == 1:
                # special case of single node
                coords = 1e-6 * np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in [0, sec.n3d() - 1]])
                lx = 2
                r = [r]

                # TODO: handle projections other than xy plane
            for ii in range(lx - 1):
                p0 = coords[ii, :2]
                p1 = coords[ii + 1, :2]
                d = p1 - p0
                dn = r[ii] * d / np.linalg.norm(d)
                n = np.array([-dn[1], dn[0]])
                pts = np.vstack((p0 + n, p1 + n, p1 - n, p0 - n))

                polys.append(pts)

    return polys


def make_biphasic_pulse(amplitude, tstart, pulsedur, trise=None):
    """
    Create pair of Vectors for biphasic (positive first) stimulus.

    Parameters
    ----------
    amplitude : float
        Amplitude of pulse in amperes.
    tstart : float
        Delay before stimulus begins in ms.
    pulsedur : float
        Duration of float in ms.
    trise : float, optional
        Pulse rise/fall time in ms.
        Defaults to pulsedur/1000 if None.

    Returns
    -------
    stimTvec : `neuron.Vector`
        Times at which stimulus is specified.
    stimVvec : `neuron.Vector`
        Amplitudes stimulus takes on.
    """
    if trise is None:
        trise = pulsedur / 1000
    dts = [0, tstart, trise, pulsedur, trise, pulsedur, trise]

    tvals = np.cumsum(dts)
    amps = amplitude * np.array([0, 0, 1, 1, -1, -1, 0])

    stimTvec = h.Vector(tvals)
    stimVvec = h.Vector(amps)

    return stimTvec, stimVvec


def make_monophasic_pulse(amplitude, tstart, pulsedur, trise=None):
    """
    Create pair of Vectors for monophasic stimulus.

    Parameters
    ----------
    amplitude : float
        Amplitude of pulse in amperes.
    tstart : float
        Delay before stimulus begins in ms.
    pulsedur : float
        Duration of float in ms.
    trise : float, optional
        Pulse rise/fall time in ms.
        Defaults to pulsedur/1000 if None.

    Returns
    -------
    stimTvec : Vector
        Times at which stimulus is specified.
    stimVvec : Vector
        Amplitudes stimulus takes on.
    """
    if trise is None:
        trise = pulsedur / 1000
    dts = [0, tstart, trise, pulsedur, trise]

    tvals = np.cumsum(dts)
    amps = amplitude * np.array([0, 0, 1, 1, 0])

    stimTvec = h.Vector(tvals)
    stimVvec = h.Vector(amps)

    return stimTvec, stimVvec


def make_interface():
    """
    Add extracellular mechanism to join NEURON and xcell.

    Returns
    -------
    float[:,3]
        Coordinates of compartment centers in meters.

    """
    h.define_shape()

    # h.nlayer_extracellular(1)
    cellcoords = []
    inds = []
    for nsec, sec in enumerate(h.allsec()):
        sec.insert("extracellular")

        ptN = sec.n3d()

        for ii in range(ptN):
            cellcoords.append([sec.x3d(ii), sec.y3d(ii), sec.z3d(ii)])
            inds.append([[nsec, ii]])

    return np.array(cellcoords) / nUnit.m


class ThresholdSim(Simulation):
    def __init__(self, name, xdom, source_amps, source_geometry, sigma=1.0):
        bbox = xdom * np.concatenate((-np.ones(3), np.ones(3)))
        super().__init__(name, bbox)
        self.sigma = sigma

        for amp, geo in zip(source_amps, source_geometry):
            self.add_current_source(amp, geometry=geo)

    def mesh_and_solve(self, depth):
        """
        Mesh and solve domain with standard xcell meshing parameters.

        Parameters
        ----------
        depth : int
            Max recursion depth
        """
        source_coords, depths, metric_coefficents = get_standard_mesh_params(self.current_sources, depth)

        self.make_adaptive_grid(
            ref_pts=source_coords, max_depth=depths, min_l0_function=general_metric, coefs=metric_coefficents
        )

        self.finalize_mesh()
        self.set_boundary_nodes()
        v = self.solve(tol=1e-9)

    def get_analytic_vals(self, coords):
        """
        Calculate the analytic voltage due to all point current sources.

        Parameters
        ----------
        coords : float[:,3]
            Cartesian coordinates at which to calculate potential

        Returns
        -------
        float[:]
            Voltage at each point in volts.
        """
        analytic_values = np.zeros(coords.shape[0])
        for src in self.current_sources:
            analytic_values += util.point_current_source_voltage(
                coords, i_source=src.value, sigma=self.sigma, source_location=src.coords
            )

        return analytic_values


class ThresholdStudy:
    def __init__(self, simulation, pulsedur=1.0, biphasic=True, viz=None):
        self.segment_coordinates = None
        self.v_external = None
        self.sim = simulation
        self.viz = viz
        self.is_biphasic = biphasic
        self.pulsedur = pulsedur

        self.cell_image = []

    def _build_neuron(self):
        """
        Placeholder: Create neural compartments in cells.

        Returns
        -------
        cell
            Cell
        """
        cell = []

        return cell

    def get_threshold(self, depth, pmin=0.0, pmax=1e2, analytic=False, strict=True):
        """
        Find activation threshold of electrode/neuron system.

        Parameters
        ----------
        depth : int
            Maximum recursion depth for mesh generation.
        pmin : float, optional
            Minimum stimulus amplitude, by default 0.
        pmax : float, optional
            Maximum stimulus amplitude, by default 1e2
        analytic : bool, optional
            Use analytic point-source approximation (True)
            or meshed geometry (False, default)
        strict : bool, optional
            Raise exception if neuron is not subthreshold
            at pmin and superthreshold at pmax; otherwise,
            return NaN for all values. Default True.

        Returns
        -------
        threshold : float
            Minimum stimulus amplitude to produce action potential
        n_elements: int
            Element count of mesh
        n_sources: int

        """
        if not analytic:
            self.sim.mesh_and_solve(depth)
            n_elements = len(self.sim.mesh.elements)
            n_sources = sum(self.sim.node_role_table == 2)
        else:
            n_sources = np.nan
            n_elements = np.nan

        if n_sources < len(self.sim.current_sources):
            n_elements = np.nan
            threshold = np.nan
            n_sources = np.nan
        else:
            if strict:
                assert self._run_trial(pmax, analytic=analytic)
                assert not self._run_trial(pmin, analytic=analytic)
            else:
                if self._run_trial(pmax, analytic=analytic) and not self._run_trial(pmin, analytic=analytic):
                    return np.nan, np.nan, np.nan

            while (pmax - pmin) > 1e-6:
                md = 0.5 * (pmin + pmax)
                # print(md)
                spike = self._run_trial(md, analytic=analytic)

                if spike:
                    pmax = md
                else:
                    pmin = md

            threshold = md
            # in amps

        if self.viz is not None and not analytic:
            self.sim.node_voltages *= threshold
            self.viz.add_simulation_data(self.sim, append=True)

        return threshold, n_elements, n_sources

    def _run_trial(self, amplitude, analytic=False):
        cell = self._build_neuron()

        if self.is_biphasic:
            tstim, vstim = make_biphasic_pulse(amplitude, 2.0, self.pulsedur)
        else:
            tstim, vstim = make_monophasic_pulse(amplitude, 2.0, self.pulsedur)

        self.tstim = tstim
        self.vstim = vstim
        tstop = 10 * max(tstim.as_numpy())

        vvecs, vmems = self._set_v_ext(analytic=analytic)

        tvec = h.Vector().record(h._ref_t)

        h.finitialize(cell.vrest)

        h.continuerun(tstop)

        spiked = len(cell.spike_times) > 0

        # cleanup NEURON objects to avoid segfault
        # should automatically happen at function return,
        # but just in case.....
        del cell

        for sec in h.allsec():
            h.delete_section(sec=sec)

        tvec.play_remove()
        tstim.play_remove()
        vstim.play_remove()
        # [v.play_remove() for v in vvecs]
        # [v.play_remove() for v in vmems]

        return spiked

    def _set_v_ext(self, analytic=False):
        setup = self.sim
        if self.v_external is None:
            if analytic:
                vext = self.sim.get_analytic_vals(self.segment_coordinates)
            else:
                vext = setup.interpolate_at_points(self.segment_coordinates)
            self.v_external = vext
        else:
            vext = self.v_external

        # tstim,vstim =xc.nrnutil.make_biphasic_pulse(k, tstart, tpulse)

        vvecs = []
        vmems = []
        for sec, V in zip(h.allsec(), vext):
            for seg in sec.allseg():
                vseg = V * nUnit.V

                vvecs.append(h.Vector(vseg * self.vstim.as_numpy()))
                vvecs[-1].play(seg.extracellular._ref_e, self.tstim, False)
                vmems.append(h.Vector().record(seg._ref_v))
                # vvecs[-1].play(seg._ref_e_extracellular, tstim, False)

        return vvecs, vmems


class RecordedCell:
    def attach_spike_detector(self, section):
        """
        Attach spike detector to section.

        Parameters
        ----------
        section : NEURON section
            Section to check for spikes.

        Returns
        -------
        None.

        """
        self._spike_detector = h.NetCon(section(0.5)._ref_v, None, sec=section)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

    def attach_membrane_recordings(self, sections=None):
        """
        Attach recorders for membrane voltage & current.

        Parameters
        ----------
        sections : NEURON section, optional
            NEURON sections to record, or all sections
            if None (default)

        Returns
        -------
        None.

        """
        if sections is None:
            sections = h.allsec()

        if "vrecs" not in dir(self):
            self.vrecs = []
        if "irecs" not in dir(self):
            self.irecs = []

        for section in sections:
            self.vrecs.append(h.Vector().record(section(0.5)._ref_v))
            self.irecs.append(h.Vector().record(section(0.5)._ref_i_membrane_))
