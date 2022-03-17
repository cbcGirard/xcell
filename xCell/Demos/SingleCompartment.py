#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:17:41 2022

@author: benoit
"""

import numpy as np
import matplotlib.pyplot as plt
import xCell

import pickle

from neuron import h#, gui
from neuron.units import ms, mV

import misc

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)


datadir='/home/benoit/smb4k/ResearchData/Results/NEURON/'#+meshtype
studyPath=datadir+'compSrc/'


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



generate=True




elementType='Admittance'

xmax=1e-4

sigma=np.ones(3)

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))
# if dual:
#     bbox+=xmax*2**(-maxdepth)


ring=Ring(N=1,r=0,stim_delay=0)
# ring=Ring(N=5,stim_delay=1)

t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(12)

# plt.scatter(t,ring.cells[0].ivec)

ivec=ring.cells[0].ivec.as_numpy()*1e-9
tvec=t.as_numpy()*1e-3
vvec=ring.cells[0].soma_v.as_numpy()*1e-3

ivec=ivec[::5]
tvec=tvec[::5]
vvec=vvec[::5]

imax=max(abs(ivec))
tmax=tvec[-1]

# tdata={
#        'x':tvec,
#        'y':vvec,
#        'ylabel':'Membrane potential',
#        'unit':'V'}
tdata={
       'x':tvec,
       'y':vvec,
       'ylabel':'Membrane\npotential',
       'unit':'V',
       'style':'dot'}

study=xCell.SimStudy(studyPath,bbox)
img=xCell.Visualizers.SingleSlice(None,study,
                                  tvec,tdata)

# err=xCell.Visualizers.SingleSlice(None, study,
#                                   tvec,tdata,
#                                   datasrc='absErr')

# img.tax.plot(tvec,vvec)
# img.tax.set_ylabel('Membrane potential [V]')


if generate:

    setup=study.newSimulation()
    setup.meshnum=-1

    setup.addCurrentSource(1.,
                           np.zeros(3),
                           1e-6)



    # maxdepth=5
    maxdepth=6

    dmin=3
    dmax=8



    lastNumEl=0
    lastI=0
    numels=[]
    Emax=[]
    Emin=[]
    Eavg=[]
    Esum=[]
    depths=[]
    densities=[]

    lists={
        'numels':[],
        'depths':[],
        'densities':[],
        'Emax':[],
        'Emin':[],
        'Eavg':[],
        'Esum':[],
        'IntAna':[],
        'IntErr':[]}

    for tval,ival,vval in zip(tvec,ivec,vvec):

        setup.currentSources[0].value=ival
        setup.currentTime=tval


        # ################ k-param strategy
        # density=0.4*(abs(ival)/imax)
        # print('density:%.2g'%density)


        ##################### Depth strategy
        maxdepth=dmin+int(np.floor((dmax-dmin)*abs(ival)/imax))
        print('depth:%d'%maxdepth)
        density=0.25


        # ################### Static mesh
        # maxdepth=6
        # density=0.2


        metric=xCell.makeExplicitLinearMetric(maxdepth, density)




        changed=setup.makeAdaptiveGrid(metric, maxdepth)



        if changed:
            setup.meshnum+=1
            setup.finalizeMesh()

            numEl=len(setup.mesh.elements)

            def bfun(coords):
                r=np.linalg.norm(coords)
                return ival/(4*np.pi*r)

            setup.setBoundaryNodes(bfun)

            v=setup.solve()
            lastI=ival
            lastNumEl=numEl
            setup.iteration+=1

            study.saveData(setup)#,baseName=str(setup.iteration))
        else:
            # vdof=setup.getDoFs()
            # v=setup.iterativeSolve(vGuess=vdof)
            v=setup.nodeVoltages*abs(ival/lastI)
            setup.nodeVoltages=v


        study.newLogEntry(['Timestep','Meshnum'],[setup.currentTime, setup.meshnum])

        setup.stepLogs=[]
        numels.append(lastNumEl)

        print('%d percent done'%(int(100*tval/tmax)))
        img.addSimulationData(setup,append=True)

        esum,err,_,sortr,r=setup.calculateErrors()
        lists['Emax'].append(max(err))
        lists['Emin'].append(min(err))
        lists['Eavg'].append(np.mean(err))
        lists['Esum'].append(esum)
        lists['densities'].append(density)
        lists['depths'].append(maxdepth)

        lists['IntAna'].append(misc.iSourceIntegral(ival,
                                                    1e-6,
                                                    np.sqrt(3)*xmax,
                                                    1))

        # r=np.linalg.norm(setup.mesh.nodeCoords,axis=1)[sortr]
        lists['IntErr'].append(np.trapz(np.abs(err[sortr]),
                                        r))
        lists['numels'].append(lastNumEl)


        # err.addSimulationData(setup,append=True)
else:
    img.getStudyData()


f,axes=plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios':[5,5,2]})


axes[0].fill_between(tvec,
                     lists['Emax'],
                     lists['Emin'],
                     color='r',alpha=0.75)
axes[0].plot(tvec,lists['Eavg'],'r')
axes[1].plot(tvec,lists['Esum'])
axes[2].plot(tvec,lists['numels'])
axes[2].xaxis.set_major_formatter(xCell.Visualizers.eform('s'))

axes[2].set_yscale('log')



ani=img.animateStudy('init',fps=30.)
# erAni=err.animateStudy('error',fps=30.)
