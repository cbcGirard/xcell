#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:54:34 2022

@author: benoit
"""

import numpy as np




def vSourceIntegral(vsrc,radius,rmax):
    return radius*vsrc*(1+np.log(rmax/radius))

def iSourceIntegral(isrc,radius,rmax,sigma):
    vsrc=isrc/(4*np.pi*sigma)
    return vSourceIntegral(vsrc, radius, rmax)

def estimatePower(voltages, edges, conductances):
    dv=voltages[edges]
    v2=np.diff(voltages,axis=1)**2
    return np.dot(v2.squeeze(),conductances)

