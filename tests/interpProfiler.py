#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:46:11 2022

@author: benoit
"""

import xcell as xc
import sys

sys.path.append("..")

from Applications import Common_nongallery
import os

here = os.path.dirname(os.path.realpath(__file__))

study, setup = Common_nongallery.makeSynthStudy(here)

max_depth = 16

Common_nongallery.runAdaptation(setup, max_depth=max_depth)

viz = xc.visualizers.SliceSet(
    None,
    study,
    prefs={
        "showError": False,
        "showInsets": False,
        "relativeError": False,
        "logScale": True,
        "show_nodes": True,
        "fullInterp": True,
    },
)

viz.add_simulation_data(setup, append=True)
viz.get_artists(0)
