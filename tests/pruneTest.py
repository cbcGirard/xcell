#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:27:55 2022

@author: benoit

"""

import numpy as np
import xCell

xmax=1e-4
bbox=xmax*np.concatenate((-np.ones(3),np.ones(3)))

setup=xCell.Simulation('', bbox)
max_depth=5

metric=xCell.makeExplicitLinearMetric(max_depth, 1)

setup.make_adaptive_grid(metric, max_depth)

setup.finalize_mesh()



metric=xCell.makeExplicitLinearMetric(max_depth, .2)

setup.make_adaptive_grid(metric, max_depth)

setup.finalize_mesh()