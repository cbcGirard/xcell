#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:14:52 2022

@author: benoit
"""

import xCell as xc
import numpy as np
import Common as com
from neuron import h#, gui
import neuron.units as nUnit

import matplotlib.pyplot as plt

nDist=3e-5


Imag=150e-6
tstop=20.
tpulse=1.
tstart=5.


study,setup=com.makeSynthStudy('NEURON/Stim',xmax=1e-4,)
setup.currentSources[0].geometry=xc.Geometry.Disk(np.zeros(3),
                                      1.25e-5,
                                      np.array([0,0,1],dtype=float))
setup.currentSources[0].value=Imag

depth=8

metric= xc.makeExplicitLinearMetric(maxdepth=depth,
                            meshdensity=0.2)

setup.makeAdaptiveGrid(metric, depth)

setup.finalizeMesh()
setup.setBoundaryNodes()
v=setup.iterativeSolve(tol=1e-9)



h.load_file('stdrun.hoc')
# h.CVode().use_fast_imem(1)

# ring=com.Ring(N=1,stim_delay=0)
cell=com.BallAndStick(1, 50., 0., 0., 0.)


h.nlayer_extracellular(1)
cellcoords=[]
inds=[]
for nsec,sec in enumerate(h.allsec()):
    sec.insert('extracellular')

    ptN=sec.n3d()

    for ii in range(ptN):
        cellcoords.append([sec.x3d(ii), sec.y3d(ii), sec.z3d(ii)])
        inds.append([[nsec,ii]])



def makeBiphasicPulse(amplitude,tstart,pulsedur,trise=None):
    if trise is None:
        trise=pulsedur/1000
    dts=[0,tstart, trise, pulsedur, trise, pulsedur, trise]

    tvals=np.cumsum(dts)
    amps=amplitude*np.array([0,0,1,1,-1, -1, 0])

    stimTvec=h.Vector(tvals)
    stimVvec=h.Vector(amps)

    return stimTvec,stimVvec


vext=setup.interpolateAt(np.array(cellcoords)/nUnit.m)


tstim,vstim =makeBiphasicPulse(-1., tstart, tpulse)

# tlist=[0., tstart, tstart+tpulse, tstart+2*tpulse, tstop]
# vlist=1e5*np.array([0.,-1.,1.,0., 0.])


# tstim=h.Vector(tlist)

vvecs=[]
vmems=[]
for sec,V in zip(h.allsec(),vext):
    for seg in sec.allseg():
        vvecs.append(h.Vector(V*nUnit.V*vstim.as_numpy()))
        vvecs[-1].play(seg.extracellular._ref_e, tstim, False)
        vmems.append(h.Vector().record(seg._ref_v))
        # vvecs[-1].play(seg._ref_e_extracellular, tstim, False)




tvec = h.Vector().record(h._ref_t)

# vref=cell.soma(0.5).extracellular._ref_e
vref=cell.soma(0.5)._ref_vext
# ve=h.Vector().record(vref,sec=cell.soma)


# g=h.Graph()
# # g.addvar('e',vref)
# # g.addvar('stim',vstim)
# pp=h.RangeVarPlot("vext",cell.dend(0),cell.soma(1))
# g.addobject(pp)
# h.graphList[0].append(g)





h.finitialize(-65*nUnit.mV)

h.continuerun(tstop)


f,axes=plt.subplots(2,sharex=True,
                    gridspec_kw={'height_ratios':[4,1]})


axes[0].plot(tvec,np.array(vmems).transpose())
axes[1].plot(tstim,np.array(vvecs).transpose())
axes[0].set_ylabel('Membrane voltage [mV]')

axes[1].set_xlabel('Time [ms]')
axes[1].set_ylabel('Extracellular Voltage [mV]')


del cell

for sec in h.allsec():
    h.delete_section(sec=sec)

# h.quit()
