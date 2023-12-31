#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:36:35 2022

@author: benoit
"""

import numpy as np
import xCell
import matplotlib.pyplot as plt


max_depth = 8
xmax = 1e-4
bbox = xmax * np.concatenate(((-np.ones(3), np.ones(3))))

sim = xCell.Simulation("", bbox)


elnums = []
kvals = np.linspace(0, 1)
for k in kvals:
    metric = xCell.makeExplicitLinearMetric(max_depth, k)
    sim.make_adaptive_grid(metric, max_depth)
    numel = len(sim.mesh.tree.get_terminal_octants())

    elnums.append(numel)


plt.scatter(kvals, elnums)
