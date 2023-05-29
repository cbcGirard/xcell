#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for interfacing with NEURON."""


from neuron import h
import neuron.units as nUnit

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from . import colors

from .xCell import Simulation, getStandardMeshParams, generalMetric
from . import util


def setVext(vext, vstim, tstim):
    vvecs = []
    vmems = []
    for sec, V in zip(h.allsec(), vext):
        for seg in sec.allseg():
            vseg = V*nUnit.V

            vvecs.append(h.Vector(vseg*vstim.as_numpy()))
            vvecs[-1].play(seg.extracellular._ref_e, tstim, False)
            vmems.append(h.Vector().record(seg._ref_v))
            # vvecs[-1].play(seg._ref_e_extracellular, tstim, False)

    return vvecs, vmems


def returnSegmentCoordinates(section, inMicrons=False):
    """
    Get geometry info at segment centers.

    Adapted from https://www.neuron.yale.edu/phpBB/viewtopic.php?p=19176#p19176

    Modified to give segment radius as well

    Parameters
    ----------
    section : NEURON section
        The section to return info about.
    inMicrons : bool, default False
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
        rad[i] = section.diam3d(i)/2

    # Compute length of each 3d segment
    for i in range(n3d):
        if i == 0:
            L[i] = 0
        else:
            L[i] = np.sqrt((x3d[i]-x3d[i-1])**2 +
                           (y3d[i]-y3d[i-1])**2 + (z3d[i]-z3d[i-1])**2)

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
        dx = section.L / (N-1)
        for n in range(N):
            if n == N-1:
                xCoord[n] = x3d[-1]
                yCoord[n] = y3d[-1]
                zCoord[n] = z3d[-1]
                rads[n] = rad[-1]
            else:
                # which idx of 3d segments are we starting at
                cIdxStart = np.where(n*dx >= cumLength)[0][-1]
                # how far along that segment is this upsampled coordinate
                cDistFrom3dStart = n*dx - cumLength[cIdxStart]
                # what's the fractional distance along this 3d segment
                cFraction3dLength = cDistFrom3dStart / L[cIdxStart+1]
                # compute x and y positions
                xCoord[n] = x3d[cIdxStart] + cFraction3dLength * \
                    (x3d[cIdxStart+1] - x3d[cIdxStart])
                yCoord[n] = y3d[cIdxStart] + cFraction3dLength * \
                    (y3d[cIdxStart+1] - y3d[cIdxStart])
                zCoord[n] = z3d[cIdxStart] + cFraction3dLength * \
                    (z3d[cIdxStart+1] - z3d[cIdxStart])
                rads[n] = rad[cIdxStart] + cFraction3dLength * \
                    (rad[cIdxStart+1] - rad[cIdxStart])

    if inMicrons:
        scale = 1.
    else:
        scale = 1e-6

    return xCoord*scale, yCoord*scale, zCoord*scale, rads*scale


def getNeuronGeometry():
    """
    Get geometric info of all compartments.

    Returns
    -------
    coords : float[:,3]
        Cartesian coordinates of compartment centers.
    rads : float[:]
        Radius of each compatment.
    isSphere : bool[:]
        Whether compartment is assumed to represent a sphere.

    """
    coords = []
    rads = []
    isSphere = []
    for sec in h.allsec():
        N = sec.n3d()-1
        x, y, z, r = returnSegmentCoordinates(sec)

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
                    sph = sec.hname().split('.')[-1] == 'soma'

                    isSphere.append(sph)
    return coords, rads, isSphere


def getMembraneCurrents():
    """
    Record total membrane current of every compartment.

    Returns
    -------
    ivecs : list of Vector
        List of vectors of membrane current.

    """
    ivecs = []
    for sec in h.allsec():
        N = sec.n3d()-1
        if N > 0:
            for ii, seg in enumerate(sec.allseg()):
                if ii == 0 or ii == N:
                    continue
                else:
                    ivec = h.Vector().record(seg._ref_i_membrane_)

                    ivecs.append(ivec)

    return ivecs


class LineDataUnits(Line2D):
    """Yoinked from https://stackoverflow.com/a/42972469"""

    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def showCellGeo(axis, polys=None, showNodes=False):
    """
    Add cell geometry to designated plot.

    Parameters
    ----------
    axis : matplotlib axis
        Axis to plot on.
    polys : List of M x 2 arrays, optional
        DESCRIPTION. The default is None, which queries NEURON for all compartments.

    Returns
    -------
    polys : TYPE
        DESCRIPTION.

    """
    shade = colors.FAINT
    if polys is None:
        polys = getCellImage()

    polycol = PolyCollection(polys, color=shade)
    axis.add_collection(polycol)
    if showNodes:
        coords = np.array([returnSegmentCoordinates(sec)
                          for sec in h.allsec()])
        axis.scatter(coords[:, 0], coords[:, 1], color='C0', marker='*')

    return polys


def getCellImage():
    """
    Get plottable representation of cell geometry.

    Returns
    -------
    polys : List of M x 2 arrays
        List of vertices for creating a PolyCollection.

    """
    tht = np.linspace(0, 2*np.pi)

    polys = []
    for sec in h.allsec():
        x, y, z, r = returnSegmentCoordinates(sec)
        coords = np.vstack((x, y, z)).transpose()

        if sec.hname().split('.')[-1] == 'soma':
            sx = x+r*np.cos(tht)
            sy = y+r*np.sin(tht)

            # axis.fill(sx, sy, color=shade)
            pts = np.vstack((sx, sy))
            polys.append(pts.transpose())

        else:
            # lx = x.shape
            lx = coords.shape[0]
            if lx == 1:
                # special case of single node
                coords = np.array([[sec.x3d(i),
                                    sec.y3d(i),
                                    sec.z3d(i)]
                                   for i in [0, sec.n3d()-1]])
                lx = 2
                r = [r]

                # TODO: handle projections other than xy plane
            for ii in range(lx-1):
                p0 = coords[ii, :2]
                p1 = coords[ii+1, :2]
                d = p1-p0
                dn = r[ii]*d/np.linalg.norm(d)
                n = np.array([-dn[1], dn[0]])
                pts = np.vstack((p0+n, p1+n, p1-n, p0-n))

                polys.append(pts)

    return polys


def makeBiphasicPulse(amplitude, tstart, pulsedur, trise=None):
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
    trise : TYPE, optional
        DESCRIPTION. Set to pulsedur/1000 if None.

    Returns
    -------
    stimTvec : Vector
        Times at which stimulus is specified.
    stimVvec : Vector
        Amplitudes stimulus takes on.
    """
    if trise is None:
        trise = pulsedur/1000
    dts = [0, tstart, trise, pulsedur, trise, pulsedur, trise]

    tvals = np.cumsum(dts)
    amps = amplitude*np.array([0, 0, 1, 1, -1, -1, 0])

    stimTvec = h.Vector(tvals)
    stimVvec = h.Vector(amps)

    return stimTvec, stimVvec


def makeMonophasicPulse(amplitude, tstart, pulsedur, trise=None):
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
    trise : TYPE, optional
        DESCRIPTION. Set to pulsedur/1000 if None.

    Returns
    -------
    stimTvec : Vector
        Times at which stimulus is specified.
    stimVvec : Vector
        Amplitudes stimulus takes on.
    """
    if trise is None:
        trise = pulsedur/1000
    dts = [0, tstart, trise, pulsedur, trise]

    tvals = np.cumsum(dts)
    amps = amplitude*np.array([0, 0, 1, 1, 0])

    stimTvec = h.Vector(tvals)
    stimVvec = h.Vector(amps)

    return stimTvec, stimVvec


def makeInterface():
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
        sec.insert('extracellular')

        ptN = sec.n3d()

        for ii in range(ptN):
            cellcoords.append([sec.x3d(ii), sec.y3d(ii), sec.z3d(ii)])
            inds.append([[nsec, ii]])

    return np.array(cellcoords)/nUnit.m


class ThresholdSim(Simulation):
    def __init__(self, name, xdom, srcAmps, srcGeometry, sigma=1.):
        bbox = xdom*np.concatenate((-np.ones(3), np.ones(3)))
        super().__init__(name, bbox)
        self.sigma = sigma

        for amp, geo in zip(srcAmps, srcGeometry):
            self.addCurrentSource(amp,
                                  coords=geo.center,
                                  geometry=geo)

    def meshAndSolve(self, depth):

        srcCoords, depths, metricCoefs = getStandardMeshParams(
            self.currentSources, depth)

        self.makeAdaptiveGrid(refPts=srcCoords, maxdepth=depths,
                              minl0Function=generalMetric, coefs=metricCoefs)

        self.finalizeMesh()
        self.setBoundaryNodes()
        v = self.iterativeSolve(tol=1e-9)

    def getAnalyticVals(self, coords):
        anaVals = np.zeros(coords.shape[0])
        for src in self.currentSources:
            anaVals += util.pointCurrentV(coords, iSrc=src.value,
                                          sigma=self.sigma,
                                          srcLoc=src.coords)

        return anaVals


class ThresholdStudy:
    def __init__(self, simulation, pulsedur=1., biphasic=True, viz=None):
        self.segCoords = None
        self.vExt = None
        self.sim = simulation
        self.viz = viz
        self.isBiphasic = biphasic
        self.pulsedur = pulsedur

        self.cellImg = []

    def _buildNeuron(self):
        cell = []

        return cell

    def getThreshold(self, depth, pmin=0, pmax=1e2, analytic=False):

        if not analytic:
            self.sim.meshAndSolve(depth)
            if self.viz is not None:
                self.viz.addSimulationData(self.sim, append=True)
            numEl = len(self.sim.mesh.elements)
            numSrc = sum(self.sim.nodeRoleTable == 2)
        else:
            numSrc = np.nan
            numEl = np.nan

        if numSrc < len(self.sim.currentSources):
            numEl = np.nan
            thresh = np.nan
            numSrc = np.nan
        else:
            assert self._runTrial(pmax, analytic=analytic)
            assert not self._runTrial(pmin, analytic=analytic)

            while (pmax-pmin) > 1e-6:
                md = 0.5*(pmin+pmax)
                # print(md)
                spike = self._runTrial(md, analytic=analytic)

                if spike:
                    pmax = md
                else:
                    pmin = md

            thresh = md
            # in amps

        return thresh, numEl, numSrc

    def _runTrial(self, amplitude, analytic=False):
        cell = self._buildNeuron()

        if self.isBiphasic:
            tstim, vstim = makeBiphasicPulse(
                amplitude, 2., self.pulsedur)
        else:
            tstim, vstim = makeMonophasicPulse(
                amplitude, 2., self.pulsedur)

        self.tstim = tstim
        self.vstim = vstim
        tstop = 10*max(tstim.as_numpy())

        vvecs, vmems = self._setVext(analytic=analytic)

        tvec = h.Vector().record(h._ref_t)

        h.finitialize(cell.vrest)

        h.continuerun(tstop)

        # memVals = np.array([v.as_numpy() for v in cell.vrecs])
        # t = tvec.as_numpy()

        # spiked = np.any(memVals > 0)
        spiked = len(cell.spike_times) > 0

        del cell

        for sec in h.allsec():
            h.delete_section(sec=sec)

        # # for vec in h.allobjects('Vector'):
        # #     vec.play_remove()
        tvec.play_remove()
        tstim.play_remove()
        vstim.play_remove()
        # [v.play_remove() for v in vvecs]
        # [v.play_remove() for v in vmems]

        return spiked

    def _setVext(self, analytic=False):

        setup = self.sim
        if self.vExt is None:
            if analytic:
                vext = self.sim.getAnalyticVals(self.segCoords)
            else:
                vext = setup.interpolateAt(self.segCoords)
            self.vExt = vext
        else:
            vext = self.vExt

        # tstim,vstim =xc.nrnutil.makeBiphasicPulse(k, tstart, tpulse)

        vvecs = []
        vmems = []
        for sec, V in zip(h.allsec(), vext):
            for seg in sec.allseg():
                vseg = V*nUnit.V

                vvecs.append(h.Vector(vseg*self.vstim.as_numpy()))
                vvecs[-1].play(seg.extracellular._ref_e, self.tstim, False)
                vmems.append(h.Vector().record(seg._ref_v))
                # vvecs[-1].play(seg._ref_e_extracellular, tstim, False)

        return vvecs, vmems


class RecordedCell():
    def attachSpikeDetector(self, section):
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
        self._spike_detector = h.NetCon(
            section(0.5)._ref_v, None, sec=section)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

    def attachMembraneRecordings(self, sections=None):
        """
        Attach recorders for membrane voltage & current.

        Parameters
        ----------
        sections : TYPE, optional
            NEURON sections to record, or all sections if None (default)

        Returns
        -------
        None.

        """
        if sections is None:
            sections = h.allsec()

        if 'vrecs' not in dir(self):
            self.vrecs = []
        if 'irecs' not in dir(self):
            self.irecs = []

        for section in sections:
            self.vrecs.append(h.Vector().record(section(0.5)._ref_v))
            self.irecs.append(h.Vector().record(section(0.5)._ref_i_membrane_))
