#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:16:59 2022

@author: benoit
"""


from neuron import h

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from .visualizers import FAINT


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
    return xCoord*1e-6, yCoord*1e-6, zCoord*1e-6, rads*1e-6


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



class LineDataUnits(Line2D):
    """
        Yoinked from https://stackoverflow.com/a/42972469
    """
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)



def showCellGeo(axis):

    tht=np.linspace(0,2*np.pi)
    shade=FAINT
    polys=[]
    for sec in h.allsec():
        x,y,z,r=returnSegmentCoordinates(sec)
        coords=np.vstack((x,y,z)).transpose()

        if sec.hname().split('.')[-1]=='soma':
            sx=x+r*np.cos(tht)
            sy=y+r*np.sin(tht)

            axis.fill(sx,sy,color=shade)
        else:
            # line=LineDataUnits(x,y, linewidth=r,color=shade)
            # axis.add_line(line)
            px=[]
            py=[]

            lx=x.shape
            if len(lx)>0:
                nseg=x.shape[0]-1
                for ii in range(nseg):
                    p0=coords[ii,:2]
                    p1=coords[ii+1,:2]
                    d=p1-p0
                    dn=r[ii]*d/np.linalg.norm(d)
                    n=np.array([-dn[1],dn[0]])
                    pts=np.vstack((p0+n, p1+n, p1-n, p0-n))
                    # a,b=np.hsplit(pts,2)

                    # px.extend(a)
                    # py.extend(b)

                    polys.append(pts)
            # else:



                # mpl.collections.LineCollection(segments, color=shade)

    polycol=PolyCollection(polys, color=shade)
    axis.add_collection(polycol)


def makeBiphasicPulse(amplitude,tstart,pulsedur,trise=None):
    if trise is None:
        trise=pulsedur/1000
    dts=[0,tstart, trise, pulsedur, trise, pulsedur, trise]

    tvals=np.cumsum(dts)
    amps=amplitude*np.array([0,0,1,1,-1, -1, 0])

    stimTvec=h.Vector(tvals)
    stimVvec=h.Vector(amps)

    return stimTvec,stimVvec

def makeMonophasicPulse(amplitude,tstart,pulsedur,trise=None):
    if trise is None:
        trise=pulsedur/1000
    dts=[0,tstart, trise, pulsedur, trise]

    tvals=np.cumsum(dts)
    amps=amplitude*np.array([0,0,1,1,0])

    stimTvec=h.Vector(tvals)
    stimVvec=h.Vector(amps)

    return stimTvec,stimVvec
