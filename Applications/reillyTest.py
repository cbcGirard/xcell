#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing Reilly 2016 params for model validity
======================================================
"""


from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

from xcell import SimStudy
from xcell.visualizers import SliceSet, pairedBar
import xcell.nrnutil as nutil

from neuron import units as nUnits
from neuron import h
from xcell.util import pointCurrentV
from xcell.geometry import Sphere
from Common import Axon10, MRG

import multiprocessing


data = read_csv('reillyThresholds.csv')

# cellLabel = 'NTC-NEUR'
cellLabel = 'WG-MRG'

sigma = 1/3  # 300 ohm-cm

xmax = 0.6
bbox = xmax*np.array([-1e-1, -1e-1, -1e-1, 1, 1e-1, 1])


ngrid = 20
xx, yy = np.meshgrid(np.linspace(bbox[0], bbox[3], ngrid),
                     np.linspace(bbox[1], bbox[4], ngrid))
gridcoords = np.vstack((xx.ravel(), yy.ravel(), np.zeros(20**2))).transpose()

# h.CVode().active(True)


# viz = None

def coordscatter(ax, coords, **kwargs):
    return ax.scatter(coords[:, 0], coords[:, 1], **kwargs)


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
        cell = MRG(0, -(nnodes//2)*1000, 0, 0, 0, axonNodes=nnodes)
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
                    srcAmps=[1., -1.],
                    srcGeometry=geom,
                    sigma=sigma)

    # study.currentSim = sim

    tpulse = rw['PulseWidth']*nUnits.s

    # stu = HocStudy(sim,
    stu = ThisStudy(sim,
                    pulsedur=tpulse,
                    # pulsedur = 1.,
                    biphasic=rw['Phases'] == 'Bi',
                    viz=viz)

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


study = SimStudy('/tmp', bbox)
study.newSimulation()
# viz = SingleSlice(None, study, )#,
viz = SliceSet(None, study,
               prefs={
                   'showError': False,
                   'showInsets': False,
                   'relativeError': False,
                   'logScale': True,
                   'showNodes': False,
                   'fullInterp': True})

myThresh = []
extraArtists = []

with multiprocessing.Pool(processes=len(data)) as pool:
    res = pool.map(run, data.iterrows())

# res = [run(it) for it in data.iterrows()]

# for ii, rw in data.iterrows():
# for itr in data.iterrows():
for dat in res:
    outsideV, planeval, gridv, thresh, segCoords = dat

    # outsideV, planeval, gridv, thresh, segCoords = run(itr)

    viz.dataSets.append({'pvals': outsideV,
                         'pcoords': segCoords[:, :-1],
                         'vArrays': planeval,
                         'meshPoints': [],
                         'sourcePoints': [],
                         })

    # viz.dataScales['vbounds'].update(planeval[0].ravel())
    viz.dataScales['vbounds'].update(gridv.ravel())

    # viz.getArtists(0)

    # # f2,a2=plt.subplots()
    a2 = viz.axes[0]

    extraArt = []

    # for src, color in zip(sim.currentSources, [(0, 0.5, 1, 1), (1, 0.5, 0, 1)]):
    #     extraArt.append(coordscatter(a2, np.array(
    #         src.coords, ndmin=2), color=color, marker='*', s=50.))

    myThresh.append(thresh*1e3)
    extraArtists.append(extraArt)
    # print('%.3g\t%.3g' % (myThresh[-1], rw[cellLabel]))


ani = viz.animateStudy(extraArtists=extraArtists)


plt.figure()
pairedBar(data[cellLabel], myThresh, ['Literature', 'Mine'])
plt.gca().set_yscale('log')
