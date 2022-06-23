#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:18:30 2022

@author: benoit
"""

import xcell
import numpy as np
from os import path

from neuron import h
from neuron.units import ms, mV


h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)


def makeSynthStudy(folderName,
                   rElec=1e-6,
                   vSource=False,
                   xmax=1e-4):
    """


    Parameters
    ----------
    folderName : TYPE
        DESCRIPTION.
    rElec : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    vSource : TYPE, optional
        DESCRIPTION. The default is False.
    xmax : TYPE, optional
        DESCRIPTION. The default is 1e-4.

    Returns
    -------
    study : TYPE
        DESCRIPTION.
    setup : TYPE
        DESCRIPTION.

    """
    datadir='/home/benoit/smb4k/ResearchData/Results'
    folder=path.join(datadir, folderName)
    bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))

    study=xcell.SimStudy(folder, bbox)

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


def runAdaptation(setup,maxdepth=8,meshdensity=0.2,metrics=None):
    if metrics is None:
        metrics=[]
        for src in setup.currentSources:
            metrics.append(xcell.makeExplicitLinearMetric(maxdepth, meshdensity,
                                                          origin=src.coords))

    setup.makeAdaptiveGrid(metrics, maxdepth)
    setup.finalizeMesh()
    setup.setBoundaryNodes()
    setup.iterativeSolve(tol=1e-9)


def setParams(testCat,testVal):

    params={
        'Mesh type':'adaptive',
        'Element type':'Admittance',
        'Vsrc?':False,
        'BoundaryFunction':None}

    params[testCat]=testVal

    return params

def pairedDepthSweep(study,depthRange, testCat,testVals, rElec=1e-6,sigma=np.ones(3)):
    lastmesh={'numel':0,'nX':0}
    meshnum=0
    for maxdepth in depthRange:
        for tstVal in testVals:
            params=setParams(testCat,tstVal)

            setup=study.newSimulation()
            setup.mesh.elementType=params['Element type']
            setup.meshtype=params['Mesh type']
            setup.meshnum=meshnum

            if params['Vsrc?']:
                setup.addVoltageSource(1,np.zeros(3),rElec)
                srcMag=1.
            else:
                srcMag=4*np.pi*sigma[0]*rElec
                setup.addCurrentSource(srcMag,np.zeros(3),rElec)

            if params['Mesh type']=='uniform':
                newNx=int(np.ceil(lastmesh['numel']**(1/3)))
                nX=newNx+newNx%2
                setup.makeUniformGrid(newNx+newNx%2)
                print('uniform, %d per axis'%nX)
            elif params['Mesh type']==r'equal $l_0$':
                setup.makeUniformGrid(lastmesh['nX'])
            else:
                metric=[xcell.makeExplicitLinearMetric(maxdepth,
                                                      meshdensity=0.2)]

                setup.makeAdaptiveGrid(metric,maxdepth)


            setup.mesh.elementType=params['Element type']
            setup.finalizeMesh()

            def boundaryFun(coord):
                r=np.linalg.norm(coord)
                return rElec/(r*np.pi*4)
            # setup.insertSourcesInMesh()

            if params['BoundaryFunction']=='Analytic':
                setup.setBoundaryNodes(boundaryFun)
            elif params['BoundaryFunction']=='Rubik0':
                setup.setBoundaryNodes(None,
                                       expand=True,
                                       sigma=1.)
            else:
                setup.setBoundaryNodes()
            # setup.getEdgeCurrents()

            # v=setup.solve()
            v=setup.iterativeSolve(None,1e-9)

            setup.applyTransforms()

            setup.getMemUsage(True)
            errEst,err,ana,_,_=setup.calculateErrors()
            FVU=xcell.misc.FVU(ana, err)

            minel=setup.mesh.getL0Min()


            print('error: %g'%errEst)

            study.newLogEntry(['Error','FVU','l0min',testCat],[errEst,FVU,minel,tstVal])
            study.saveData(setup)
            if params['Mesh type']=='adaptive':
                lastmesh={'numel':len(setup.mesh.elements),
                          'nX':setup.ptPerAxis-1}



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

    def __init__(self, gid, x, y, z, theta):
        super().__init__(gid, x, y, z, theta)
        r=self.soma.diam/2
        dx=r*np.cos(theta)
        dy=r*np.sin(theta)

        self._set_position(self.x-dx,self.y-dy, self.z)

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

