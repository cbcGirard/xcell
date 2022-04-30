#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:16:59 2022

@author: benoit
"""


from neuron import h

import numpy as np


def returnSegmentCoordinates(section):
    """
    Get geometry info at segment centers.

    Adapted from https://www.neuron.yale.edu/phpBB/viewtopic.php?p=19176#p19176

    Modified to give segment radius as well

    Parameters
    ----------
    section : TYPE
        DESCRIPTION.

    Returns
    -------
    xCoord : TYPE
        DESCRIPTION.
    yCoord : TYPE
        DESCRIPTION.
    zCoord : TYPE
        DESCRIPTION.

    """
    # Get section 3d coordinates and put in numpy array
    n3d = section.n3d()
    x3d = np.empty(n3d)
    y3d = np.empty(n3d)
    z3d = np.empty(n3d)
    rad = np.empty(n3d)
    L = np.empty(n3d)
    for i in range(n3d):
        x3d[i]=section.x3d(i)
        y3d[i]=section.y3d(i)
        z3d[i]=section.z3d(i)
        rad[i]=section.diam3d(i)/2

    # Compute length of each 3d segment
    for i in range(n3d):
        if i==0:
            L[i]=0
        else:
            L[i]=np.sqrt((x3d[i]-x3d[i-1])**2 + (y3d[i]-y3d[i-1])**2 + (z3d[i]-z3d[i-1])**2)

    # Get cumulative length of 3d segments
    cumLength = np.cumsum(L)

    N = section.nseg

    if N==1:
        #special case of single segment, e.g. a soma
        xCoord=np.array(x3d[1])
        yCoord=np.array(y3d[1])
        zCoord=np.array(z3d[1])
        rads=np.array(rad[1])

    else:

        # Now upsample coordinates to segment locations
        xCoord = np.empty(N)
        yCoord = np.empty(N)
        zCoord = np.empty(N)
        rads=np.empty(N)
        dx = section.L / (N-1)
        for n in range(N):
            if n==N-1:
                xCoord[n]=x3d[-1]
                yCoord[n]=y3d[-1]
                zCoord[n]=z3d[-1]
                rads[n]=rad[-1]
            else:
                cIdxStart = np.where(n*dx >= cumLength)[0][-1] # which idx of 3d segments are we starting at
                cDistFrom3dStart = n*dx - cumLength[cIdxStart] # how far along that segment is this upsampled coordinate
                cFraction3dLength = cDistFrom3dStart / L[cIdxStart+1] # what's the fractional distance along this 3d segment
                # compute x and y positions
                xCoord[n] = x3d[cIdxStart] + cFraction3dLength*(x3d[cIdxStart+1] - x3d[cIdxStart])
                yCoord[n] = y3d[cIdxStart] + cFraction3dLength*(y3d[cIdxStart+1] - y3d[cIdxStart])
                zCoord[n] = z3d[cIdxStart] + cFraction3dLength*(z3d[cIdxStart+1] - z3d[cIdxStart])
                rads[n] = rad[cIdxStart] + cFraction3dLength*(rad[cIdxStart+1] - rad[cIdxStart])
    return xCoord, yCoord, zCoord,rads


def getNeuronGeometry():

    ivecs=[]
    coords=[]
    rads=[]
    isSphere=[]
    for sec in h.allsec():
        N=sec.n3d()-1
        x,y,z,r=returnSegmentCoordinates(sec)
        r=r*1e-6
        coord=np.vstack((x,y,z)).transpose()
        coords.extend(coord*1e-6)
        if coord.shape[0]==1:
            rads.append(r.tolist())
        else:
            rads.extend(r.tolist())
        if N>0:
            for ii,seg in enumerate(sec.allseg()):
                if ii==0 or ii==N:
                    continue
                else:
                    ivec=h.Vector().record(seg._ref_i_membrane_)


                    # where=ii/N
                    # x=sec.x3d(ii)
                    # y=sec.y3d(ii)
                    # z=sec.z3d(ii)
                    # rad=sec.diam3d(ii)

                    # coords.append(np.array([x,y,z]))
                    # rads.append(rad)
                    ivecs.append(ivec)

                    sph= sec.hname().split('.')[-1]=='soma'

                    isSphere.append(sph)
    return ivecs, isSphere, coords, rads

