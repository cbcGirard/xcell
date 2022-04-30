#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:14:52 2022

@author: benoit
"""

import xCell as xc
import numpy as np
import Common as com
from neuron import h, gui
from neuron.units import ms, mV


import matplotlib.pyplot as plt

nDist=3e-5


study,setup=com.makeSynthStudy('NEURON/Stim',xmax=1e-4,)
setup.currentSources[0].geometry=xc.Geometry.Disk(np.zeros(3),
                                      1.25e-5,
                                      np.array([0,0,1],dtype=float))
setup.currentSources[0].value=1.

depth=8

metric= xc.makeExplicitLinearMetric(maxdepth=depth,
                            meshdensity=0.2)

setup.makeAdaptiveGrid(metric, depth)

setup.finalizeMesh()
setup.setBoundaryNodes()
v=setup.iterativeSolve(tol=1e-9)

v0=setup.interpolateAt(np.array([nDist,0,0],ndmin=2))


h.load_file('stdrun.hoc')
h.load_file('interpxyz.hoc')
h.load_file('setpointers.hoc')
# try:
#     h.nrn_load_dll('x86_64/libnrnmech.so')
# except:
#     pass
# h.CVode().use_fast_imem(1)

# ring=com.Ring(N=1,stim_delay=0)
cell=com.BallAndStick(1, 50., 0., 0., 0.)


# h.nlayer_extracellular(1)
cellcoords=[]
inds=[]
for nsec,sec in enumerate(h.allsec()):
    sec.insert('xtra')

    ptN=sec.n3d()
    ii=ptN//2
    cellcoords.append([sec.x3d(ii), sec.y3d(ii), sec.z3d(ii)])




def makeBiphasicPulse(amplitude,tstart,pulsedur,trise=None):
    if trise is None:
        trise=pulsedur/1000
    dts=[0,tstart, trise, pulsedur, trise, pulsedur, trise]

    tvals=np.cumsum(dts)
    amps=amplitude*np.array([0,0,1,1,-1, -1, 0])

    stimTvec=h.Vector(tvals)
    stimVvec=h.Vector(amps)

    return stimTvec,stimVvec


vext=setup.interpolateAt(1e-6*np.array(cellcoords))

tstop=100.
tpulse=10.
tstart=2.

tstim,vstim =makeBiphasicPulse(-1e3, tstart, tpulse)


for sec,v in zip(h.allsec(),vext):
    sec.rx1_xtra=v*1e3



h.init()
h.finitialize(-65*mV)
h.fcurrent()


tvec = h.Vector().record(h._ref_t)

vstim.play(h._ref_is1_xtra, tstim, True)

# h.continuerun(tstop)
h.run()

plt.plot(tvec,cell.soma_v)
plt.plot(tstim,vstim)




# while h.t < tstop:
#     if pulseStart<= h.t <= pulseEnd:
#         for s,v in zip(h.allsec(),vext):
#             s.extracellular.

# ax=xc.Visualizers.showMesh(setup)

# srcpts=setup.mesh.nodeCoords[setup.nodeRoleTable==2]
# xc.Visualizers.showNodes3d(ax, srcpts, np.ones(srcpts.shape[0]))


# P=xc.Visualizers.LogError(None, study)
# P.addSimulationData(setup,True)
# P.getArtists(0)