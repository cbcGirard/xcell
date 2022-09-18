#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:19:45 2022

@author: benoit
"""

import os

folder='Quals/Threshold/'

for d in range(50,200,50):


        cmd='python MonoStim.py -Y %d -f %s -vl'%(d, folder)
        # print(cmd)
        os.system(cmd)


os.system('python MonoStim.py -Y %d -f %s -vlS'%(200,folder))