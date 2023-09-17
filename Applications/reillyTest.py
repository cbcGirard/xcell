#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing Reilly 2016 params for model validity
======================================================
"""


from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

from xcell import Study
from xcell.visualizers import SliceSet, paired_bars
import xcell.nrnutil as nutil

from neuron import units as nUnits
from neuron import h
from xcell.util import point_current_source_voltage
from xcell.geometry import Sphere
from Common import Axon10, MRG

import multiprocessing


data = read_csv("reillyThresholds.csv")

# cellLabel = 'NTC-NEUR'
cellLabel = "WG-MRG"

sigma = 1 / 3  # 300 ohm-cm

xmax = 0.6
bbox = xmax * np.array([-1e-1, -1e-1, -1e-1, 1, 1e-1, 1])


ngrid = 20
xx, yy = np.meshgrid(np.linspace(bbox[0], bbox[3], ngrid), np.linspace(bbox[1], bbox[4], ngrid))
gridcoords = np.vstack((xx.ravel(), yy.ravel(), np.zeros(20**2))).transpose()

# h.CVode().active(True)


# viz = None


def coordscatter(ax, coords, **kwargs):
    return ax.scatter(coords[:, 0], coords[:, 1], **kwargs)


class HalfPlane(nutil.ThresholdSim):
    def get_analytic_vals(self, coords):
        analytic_values = np.zeros(coords.shape[0])
        for src in self.current_sources:
            analytic_values += 2 * point_current_source_voltage(
                coords, i_source=src.value, sigma=self.sigma, source_location=src.coords
            )

        return analytic_values


class ThisStudy(nutil.ThresholdStudy):
    def _build_neuron(self):
        nnodes = 101
        cell = MRG(0, -(nnodes // 2) * 1000, 0, 0, 0, axonNodes=nnodes)
        # cell = Axon10(1, -(nnodes//2)*1000, 0, 0, nnodes)

        self.segment_coordinates = nutil.make_interface()

        # optional visualization
        if self.viz is not None:
            # viz.add_simulation_data(setup,append=True)
            self.cell_image = nutil.show_cell_geo(self.viz.axes[0])

        return cell


class Hcell(nutil.RecordedCell):
    def __init__(self):
        nodes = []
        internodes = []
        allsec = []

        h.load_file(1, "estimsurvey/axon10.hoc")

        for sec in h.allsec():
            allsec.append(sec)
            if "internode" in sec.name():
                internodes.append(sec)
            else:
                nodes.append(sec)

        self.nodes = nodes
        self.internodes = internodes
        self.nnodes = len(nodes)

        self.vrest = -70

        self.attach_spike_detector(self.nodes[self.nnodes // 2])
        self.attach_membrane_recordings()


class HocStudy(nutil.ThresholdStudy):
    def _build_neuron(self):
        cell = Hcell()

        self.segment_coordinates = nutil.make_interface()

        return cell

    def get_threshold(self, depth, pmin=0, pmax=1e6, analytic=False):
        threshold, n_elements, n_sources = super().get_threshold(
            depth=depth, pmin=pmin, pmax=pmax, analytic=analytic
        )
        # h.quit()

        return threshold, n_elements, n_sources


def run(iterator):
    ii, rw = iterator
    coordA = np.array([rw["xA"], rw["yA"], 0])
    coordB = np.array([rw["xC"], rw["yC"], 0])

    geom = [Sphere(c, 0) for c in [coordA, coordB]]

    sim = HalfPlane("test", xdom=xmax, source_amps=[1.0, -1.0], source_geometry=geom, sigma=sigma)

    # study.current_simulation = sim

    tpulse = rw["PulseWidth"] * nUnits.s

    # stu = HocStudy(sim,
    stu = ThisStudy(
        sim,
        pulsedur=tpulse,
        # pulsedur = 1.,
        biphasic=rw["Phases"] == "Bi",
        viz=viz,
    )

    # h.CVode().active(True)
    if tpulse < 1.0:
        h.dt = 5e-3
    else:
        h.dt = 0.025

    threshold, _, _ = stu.get_threshold(0, analytic=True)

    gridv = threshold * sim.get_analytic_vals(gridcoords)
    planeval = [gridv.reshape((ngrid, ngrid))]

    outsideV = stu.v_external * threshold
    segment_coordinates = stu.segment_coordinates

    return outsideV, planeval, gridv, threshold, segment_coordinates


study = Study("/tmp", bbox)
study.new_simulation()
# viz = SingleSlice(None, study, )#,
viz = SliceSet(
    None,
    study,
    prefs={
        "showError": False,
        "showInsets": False,
        "relativeError": False,
        "logScale": True,
        "show_nodes": False,
        "fullInterp": True,
    },
)

myThresh = []
extra_artists = []

with multiprocessing.Pool(processes=len(data)) as pool:
    res = pool.map(run, data.iterrows())

# res = [run(it) for it in data.iterrows()]

# for ii, rw in data.iterrows():
# for itr in data.iterrows():
for dat in res:
    outsideV, planeval, gridv, threshold, segment_coordinates = dat

    # outsideV, planeval, gridv, threshold, segment_coordinates = run(itr)

    viz.datasets.append(
        {
            "pvals": outsideV,
            "pcoords": segment_coordinates[:, :-1],
            "vArrays": planeval,
            "meshPoints": [],
            "sourcePoints": [],
        }
    )

    # viz.data_scales['vbounds'].update(planeval[0].ravel())
    viz.data_scales["vbounds"].update(gridv.ravel())

    # viz.get_artists(0)

    # # f2,a2=plt.subplots()
    a2 = viz.axes[0]

    extraArt = []

    # for src, color in zip(sim.current_sources, [(0, 0.5, 1, 1), (1, 0.5, 0, 1)]):
    #     extraArt.append(coordscatter(a2, np.array(
    #         src.coords, ndmin=2), color=color, marker='*', s=50.))

    myThresh.append(threshold * 1e3)
    extra_artists.append(extraArt)
    # print('%.3g\t%.3g' % (myThresh[-1], rw[cellLabel]))


ani = viz.animate_study(extra_artists=extra_artists)


plt.figure()
paired_bars(data[cellLabel], myThresh, ["Literature", "Mine"])
plt.gca().set_yscale("log")
