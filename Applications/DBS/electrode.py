#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
# %%
import xcell
import pyvista as pv
import pickle
import os
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm

from makeStim import getSignals
from itertools import chain


def makeDBSElectrode(dbody=1.3e-3, bodyL=0.05,
                     orientation=np.array([1., 0.1, 0.]),
                     tipPt=1e-3*np.array([-9., 1., 0]),

                     wmacro=1.6e-3, pitchMacro=5e-3, nmacro=4,
                     dmicro=40e-6,

                     nmicro=15, microRows=6, microCols=4,
                     dualRows=True,
                     microStyle='disk'
                     ):

    # nmicro 10-24
    # nmacro 4-6
    nmicro = microRows*microCols
    body = xcell.geometry.Cylinder(center=tipPt + bodyL*orientation/2,
                                   radius=dbody/2,
                                   length=bodyL,
                                   axis=orientation
                                   )

    microElectrodes = []
    macroElectrodes = []
    # Generate microelectrodes
    for ii in range(microRows):
        if dualRows:
            slot, rem = divmod(ii, 2)
            scoot = 0.125*(-1)**rem
            shift = slot+scoot+1
        else:
            slot = ii+0.5
            shift = ii+0.5

        rowpt = tipPt+(shift)*pitchMacro*orientation

        for jj in range(microCols):
            rot = Rotation.from_rotvec(orientation*2*jj/microCols*np.pi)

            if microStyle == 'cylinder':
                microOrientation = rot.apply(
                    0.5*(dbody-dmicro)*np.array([0., 0., 1.]))
                geo = xcell.geometry.Cylinder(length=dmicro,
                                              center=rowpt+microOrientation,
                                              radius=dmicro/2, axis=microOrientation)

            else:

                microOrientation = rot.apply(
                    0.5*(dbody)*np.array([0., 0., 1.]))
                if microStyle == 'sphere':
                    geo = xcell.geometry.Sphere(
                        center=rowpt+microOrientation,
                        radius=dmicro/2)
                else:

                    geo = xcell.geometry.Disk(tol=0.5,
                                              center=rowpt+microOrientation,
                                              radius=dmicro/2, axis=microOrientation)

            microElectrodes.append(geo)

    # channels = pickle.load(open('pattern.xstim', 'rb'))

    # Generate macroelectrodes (bands)
    for ii in range(nmacro):
        if dualRows:
            shift = ii+0.5
        else:
            shift = ii+1
        pt = tipPt+shift*pitchMacro*orientation

        geo = xcell.geometry.Cylinder(pt, dbody/2, wmacro, orientation)
        macroElectrodes.append(geo)

    return body, microElectrodes, macroElectrodes


def insertInRegion(regions, body, microElectrodes, macroElectrodes):
    regions['Insulators'].append(xcell.geometry.toPV(body))
    for geo in chain(microElectrodes, macroElectrodes):
        regions['Electrodes'].append(xcell.geometry.toPV(geo))


def addStandardStim(simulation, microElectrodes, macroElectrodes):
    nmicro = len(microElectrodes)
    channels = getSignals(nmicro)

    for geo, ch in zip(microElectrodes, channels):
        simulation.addCurrentSource(ch, coords=geo.center,
                                    geometry=geo)

    for geo in macroElectrodes:
        simulation.addCurrentSource(xcell.signals.Signal(0),
                                    coords=geo.center,
                                    geometry=geo)


def addUnityStim(simulation, microElectrodes, macroElectrodes, whichActive, amplitude=1e-6):
    for ii, geo in enumerate(chain(microElectrodes, macroElectrodes)):
        srcVal = xcell.signals.Signal(amplitude*float(ii == whichActive))
        simulation.addCurrentSource(srcVal, coords=geo.center,
                                    geometry=geo)
