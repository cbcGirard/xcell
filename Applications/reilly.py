#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:18:15 2022

@author: benoit
"""

import xcell.nrnutil as nutil

from neuron import units as nUnits
from neuron import h
from xcell.util import point_current_source_voltage
from xcell.geometry import Sphere
from Common import Axon10, MRG

import numpy as np

sigma = 1 / 3  # 300 ohm-cm

xmax = 0.6
bbox = xmax * np.array([-1e-1, -1e-1, -1e-1, 1, 1e-1, 1])
ngrid = 20
xx, yy = np.meshgrid(np.linspace(bbox[0], bbox[3], ngrid), np.linspace(bbox[1], bbox[4], ngrid))
gridcoords = np.vstack((xx.ravel(), yy.ravel(), np.zeros(20**2))).transpose()


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
        cell = MRG(0, 0, 0, 0, 0)
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

    sim = HalfPlane("test", xdom=xmax, source_amps=[-1.0, 1.0], source_geometry=geom, sigma=sigma)

    # study.current_simulation = sim

    tpulse = rw["PulseWidth"] * nUnits.s

    # stu = HocStudy(sim,
    stu = ThisStudy(
        sim,
        pulsedur=tpulse,
        # pulsedur = 1.,
        biphasic=rw["Phases"] == "Bi",
    )  # ,
    # viz=viz)

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
