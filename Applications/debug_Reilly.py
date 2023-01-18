#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call .hoc models
=====================

Directly calling NTC model from Reilly study
"""

import neuron as nrn
from neuron import h

import Common as comcom
import xcell.nrnutil as nutil
from pandas import read_csv


nrn.load_mechanisms('estimsurvey/', False)
data = read_csv('reillyThresholds.csv')


class ThisStudy(nutil.ThresholdStudy):
    def _buildNeuron(self):
        h.load_file('estimsurvey/axon10.hoc')

        self.segCoords = nutil.makeInterface()


sigma = 1/3
xmax = 0.6


sim = nutil.ThresholdSim('test', xdom=xmax,
                         srcAmps=[-1., 1.],
                         srcGeometry=geom,
                         sigma=sigma)
