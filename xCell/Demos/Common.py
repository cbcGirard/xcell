#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:18:30 2022

@author: benoit
"""

import xCell
import numpy as np
from os import path

def makeSynthStudy(folderName,
                   rElec=1e-6,
                   vSource=False,
                   xmax=1e-4):
    datadir='/home/benoit/smb4k/ResearchData/Results'
    folder=path.join(datadir, folderName)
    bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))

    study=xCell.SimStudy(folder, bbox)
    
    setup=study.newSimulation()
    
    if vSource:
        setup.addVoltageSource(1.,
                               coords=np.zeros(3),
                               radius=rElec)
    else:
        setup.addCurrentSource(4*np.pi*rElec,
                               coords=np.zeros(3),
                               radius=rElec)
        
    return study,setup