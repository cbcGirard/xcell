#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:19:45 2022

Quick and dirty replotting of threshold tests

@author: benoit
"""

import os

# folder='Ball2mm/'

# x=2000

color=''
# color='l'

folder = 'axon5cm/'
x=10000

ymax=250

ystep=50

for d in range(ystep,ymax,ystep):


        cmd='python MonoStim.py -Y %d -x %d -f %s -v'%(d, x, folder)+color
        # print(cmd)
        os.system(cmd)


os.system('python MonoStim.py -Y %d -x %d -f %s -vS'%(ymax,x,folder)+color)
