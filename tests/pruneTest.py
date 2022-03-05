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
maxdepth=5

metric=xCell.makeExplicitLinearMetric(maxdepth, 1)

setup.makeAdaptiveGrid(metric, maxdepth)

setup.finalizeMesh()



metric=xCell.makeExplicitLinearMetric(maxdepth, .2)

setup.makeAdaptiveGrid(metric, maxdepth)

setup.finalizeMesh()