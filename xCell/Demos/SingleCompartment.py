#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:17:41 2022

@author: benoit
"""

import numpy as np
import matplotlib.pyplot as plt
import xCell



from neuron import h, gui
from neuron.units import ms, mV

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)


datadir='/home/benoit/smb4k/ResearchData/Results/studyTst/'#+meshtype
studyPath=datadir+'NEURON/'


class Cell:
    def __init__(self, gid, x, y, z, theta):
        self._gid = gid
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z = 0
        h.define_shape()
        self._rotate_z(theta)
        self._set_position(x, y, z)
        
        # everything below here in this method is NEW
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)
        
        self._ncs = []
        
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)
        self.ivec=h.Vector().record(self.soma(0.5)._ref_i_membrane_)
        
    def __repr__(self):
        return '{}[{}]'.format(self.name, self._gid)
    
    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(i,
                               x - self.x + sec.x3d(i),
                               y - self.y + sec.y3d(i),
                               z - self.z + sec.z3d(i),
                              sec.diam3d(i))
        self.x, self.y, self.z = x, y, z
        
    def _rotate_z(self, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))

class BallAndStick(Cell):
    name = 'BallAndStick'
    
    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')                                          
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003    # Leak conductance in S/cm2
            seg.hh.el = -54.3     # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert('pas')                 
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65    # Leak reversal potential mV

        # NEW: the synapse
        self.syn = h.ExpSyn(self.dend(0.5))
        self.syn.tau = 2 * ms
        
        
class Ring:
    """A network of *N* ball-and-stick cells where cell n makes an
    excitatory synapse onto cell n + 1 and the last, Nth cell in the
    network projects to the first cell.
    """
    def __init__(self, N=5, stim_w=0.04, stim_t=4, stim_delay=1, syn_w=0.01, syn_delay=5, r=50):
        """
        :param N: Number of cells.
        :param stim_w: Weight of the stimulus
        :param stim_t: time of the stimulus (in ms)
        :param stim_delay: delay of the stimulus (in ms)
        :param syn_w: Synaptic weight
        :param syn_delay: Delay of the synapse
        :param r: radius of the network
        """ 
        self._syn_w = syn_w
        self._syn_delay = syn_delay
        self._create_cells(N, r)
        self._connect_cells()
        # add stimulus
        self._netstim = h.NetStim()
        self._netstim.number = 1
        self._netstim.start = stim_t
        self._nc = h.NetCon(self._netstim, self.cells[0].syn)
        self._nc.delay = stim_delay
        self._nc.weight[0] = stim_w
    
    def _create_cells(self, N, r):
        self.cells = []
        for i in range(N):
            theta = i * 2 * h.PI / N
            self.cells.append(BallAndStick(i, h.cos(theta) * r, h.sin(theta) * r, 0, theta))
    
    def _connect_cells(self):
        for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
            nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            source._ncs.append(nc)    
        


ring=Ring(N=1,r=0)




t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(25)

plt.scatter(t,ring.cells[0].ivec)



elementType='Admittance'

xmax=1e-4
maxdepth=8

sigma=np.ones(3)


vMode=False
showGraphs=False
generate=False
saveGraphs=False

dual=True
regularize=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))
# if dual:
#     bbox+=xmax*2**(-maxdepth)


study=xCell.SimStudy(studyPath,bbox)
setup=study.newSimulation()

setup.addCurrentSource(1.,
                       np.zeros(3), 
                       1e-6)

# img=xCell.Visualizers.SliceSet(plt.figure(),study)

ivec=ring.cells[0].ivec.as_numpy()
tvec=t.as_numpy()
vvec=ring.cells[0].soma_v.as_numpy()

imax=max(abs(ivec))
tmax=tvec[-1]


lastNumEl=0
lastI=0
for tval,ival,vval in zip(tvec,ivec,vvec):
    
    setup.currentSources[0].value=ival
    setup.currentTime=tval
    
    if abs((lastI-ival)/ival)>0.5:
        
        k=2**(-maxdepth*0.2)*(1-abs(ival)/imax)
        
        print(k)
        
        def metric(coord,kest=k):
            r=np.linalg.norm(coord)
            val=kest*r
            
            if (r+val)<1e-7:
                val=1e-7
            return val
        
        
        setup.makeAdaptiveGrid(metric, maxdepth)
        setup.finalizeMesh()
        
        numEl=len(setup.mesh.elements)
        
        setup.setBoundaryNodes()
        
        v=setup.iterativeSolve()
        lastI=ival
        lastNumEl=numEl
    
    else:
        # vdof=setup.getDoFs()
        # v=setup.iterativeSolve(vGuess=vdof)
        v=setup.nodeVoltages*(ival/lastI)
        setup.nodeVoltages=v
    
    study.newLogEntry()
    study.saveData(setup,'_'+str(setup.iteration))
    
    setup.iteration+=1
    setup.stepLogs=[]
    

    print('%d percent done'%(int(100*tval/tmax)))
    # img.addSimulationData(setup)
    
    
# art=img.getArtists(0)

# img.animateStudy(None,art)