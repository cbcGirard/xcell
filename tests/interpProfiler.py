#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:46:11 2022

@author: benoit
"""

import xcell as xc
import sys

sys.path.append('..')

from Applications import Common
import os

here=os.path.dirname(os.path.realpath(__file__))

study,setup=Common.makeSynthStudy(here)

maxdepth=16

Common.runAdaptation(setup,maxdepth=maxdepth)

viz=xc.visualizers.SliceSet(None, study,
                                prefs={
                                    'showError':False,
                                    'showInsets':False,
                                    'relativeError':False,
                                    'logScale':True,
                                    'showNodes':True,
                                    'fullInterp':True})

viz.addSimulationData(setup,append=True)
viz.getArtists(0)