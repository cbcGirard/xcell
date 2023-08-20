#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the full set of data for one neuron's thresholdold study
"""

import os

# color = ''
color = 'l'

celltype = 'MRG'
# celltype = 'Ax10'
x = 10000
# celltype = 'ball'
# x = 1000

folder = '"tthresholdold-%s"' % celltype

ymax = 150

ystep = 25

for d in range(0, ymax, ystep):

    cmd = 'python Thresholds.py -C %s -Y %d -x %d -f %s -v' % (
        celltype, d, x, folder)+color
    # print(cmd)
    os.system(cmd)


os.system('python Thresholds.py -C %s -Y %d -x %d -f %s -vS' %
          (celltype, ymax, x, folder)+color)
