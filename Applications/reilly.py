#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:18:15 2022

@author: benoit
"""

import xcell.nrnutil as nutil

from neuron import units as nUnits
from neuron import h
from xcell.util import pointCurrentV
from xcell.geometry import Sphere
from Common import Axon10, MRG

import numpy as np

sigma = 1/3  # 300 ohm-cm

xmax = 0.6
bbox = xmax*np.array([-1e-1, -1e-1, -1e-1, 1, 1e-1, 1])
ngrid = 20
xx, yy = np.meshgrid(np.linspace(bbox[0], bbox[3], ngrid),
                     np.linspace(bbox[1], bbox[4], ngrid))
gridcoords = np.vstack((xx.ravel(), yy.ravel(), np.zeros(20**2))).transpose()


class HalfPlane(nutil.ThresholdSim):
    def getAnalyticVals(self, coords):
        anaVals = np.zeros(coords.shape[0])
        for src in self.currentSources:
            anaVals += 2*pointCurrentV(coords,
                                       iSrc=src.value,
                                       sigma=self.sigma,
                                       srcLoc=src.coords)

        return anaVals


class ThisStudy(nutil.ThresholdStudy):
    def _buildNeuron(self):
        nnodes = 101
        cell = MRG(0, 0, 0, 0, 0)
        # cell = Axon10(1, -(nnodes//2)*1000, 0, 0, nnodes)

        self.segCoords = nutil.makeInterface()

        # optional visualization
        if self.viz is not None:
            # viz.addSimulationData(setup,append=True)
            self.cellImg = nutil.showCellGeo(self.viz.axes[0])

        return cell


class Hcell(nutil.RecordedCell):
    def __init__(self):
        nodes = []
        internodes = []
        allsec = []

        h.load_file(1, 'estimsurvey/axon10.hoc')

        for sec in h.allsec():
            allsec.append(sec)
            if 'internode' in sec.name():
                internodes.append(sec)
            else:
                nodes.append(sec)

        self.nodes = nodes
        self.internodes = internodes
        self.nnodes = len(nodes)

        self.vrest = -70

        self.attachSpikeDetector(self.nodes[self.nnodes//2])
        self.attachMembraneRecordings()


class HocStudy(nutil.ThresholdStudy):
    def _buildNeuron(self):
        cell = Hcell()

        self.segCoords = nutil.makeInterface()

        return cell

    def getThreshold(self, depth, pmin=0, pmax=1e6, analytic=False):
        thresh, numEl, numSrc = super().getThreshold(depth=depth, pmin=pmin,
                                                     pmax=pmax, analytic=analytic)
        # h.quit()

        return thresh, numEl, numSrc


def run(iterator):
    ii, rw = iterator
    coordA = np.array([rw['xA'], rw['yA'], 0])
    coordB = np.array([rw['xC'], rw['yC'], 0])

    geom = [Sphere(c, 0) for c in [coordA, coordB]]

    sim = HalfPlane('test',
                    xdom=xmax,
                    srcAmps=[-1., 1.],
                    srcGeometry=geom,
                    sigma=sigma)

    # study.currentSim = sim

    tpulse = rw['PulseWidth']*nUnits.s

    # stu = HocStudy(sim,
    stu = ThisStudy(sim,
                    pulsedur=tpulse,
                    # pulsedur = 1.,
                    biphasic=rw['Phases'] == 'Bi')  # ,
    # viz=viz)

    # h.CVode().active(True)
    if tpulse < 1.:
        h.dt = 5e-3
    else:
        h.dt = .025

    thresh, _, _ = stu.getThreshold(0, analytic=True)

    gridv = thresh*sim.getAnalyticVals(gridcoords)
    planeval = [gridv.reshape((ngrid, ngrid))]

    outsideV = stu.vExt*thresh
    segCoords = stu.segCoords

    return outsideV, planeval, gridv, thresh, segCoords
