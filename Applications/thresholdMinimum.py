#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:55:28 2022

@author: benoit
"""

from neuron import h


h.load_file('stdrun.hoc')

h.nrn_load_dll('estimsurvey/x86_64/libnrnmech.so')
h.load_file('estimsurvey/initA.hoc')



# for ii in range(1,9):
#     h.protocol(ii)



    # #set rx
#     h.stimamp=0
#     thresh=h.threshold(h._ref_stimamp)