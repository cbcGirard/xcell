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
import time

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)


datadir='/home/benoit/smb4k/ResearchData/Results/NEURON/'#+meshtype
studyPath=datadir+'tmp/coarse/'


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



# generate=True
# vids=True
# post=True


generate=False
vids=False
post=True

strat='depth'


elementType='Admittance'

xmax=1e-3

#2.6 ohm-m .384/m
#6.4 ohm-m, 0.156
sigma=0.156*np.ones(3)

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

ivec=-ring.cells[0].ivec.as_numpy()*1e-9
tvec=t.as_numpy()*1e-3
vvec=ring.cells[0].soma_v.as_numpy()*1e-3

# ivec=ivec[::5]
# tvec=tvec[::5]
# vvec=vvec[::5]

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

if vids:
    img=xCell.Visualizers.SingleSlice(None,study,
                                      tvec,tdata)

    err=xCell.Visualizers.SingleSlice(None, study,
                                      tvec,tdata,
                                      datasrc='absErr')

# img.tax.plot(tvec,vvec)
# img.tax.set_ylabel('Membrane potential [V]')


if generate:

    setup=study.newSimulation()
    setup.meshnum=-1

    setup.addCurrentSource(1.,
                           np.zeros(3),
                           1e-6*ring.cells[0].soma.diam3d(0)/2)



    # maxdepth=5
    maxdepth=6

    dmin=6
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
        'IntErr':[],
        'FVU':[],
        'FVU2':[],
        'vol':[],
        'volAna':[],
        'volErr':[],
        'vMax':[]
        }


    errdicts=[]

    for tval,ival,vval in zip(tvec,ivec,vvec):

        t0=time.monotonic()
        setup.currentSources[0].value=-ival
        setup.currentTime=tval


        if strat=='k':
            ################ k-param strategy
            density=0.4*(abs(ival)/imax)
            print('density:%.2g'%density)

        elif strat=='depth' or strat=='d2':
            ##################### Depth strategy
            scale=(dmax-dmin)*abs(ival)/imax
            dint,dfrac=divmod(scale, 1)
            maxdepth=dmin+int(dint)
            print('depth:%d'%maxdepth)

            # density=0.25
            if strat=='d2':
                density=0.5
            else:
                density=0.2#+0.2*dfrac

        elif strat=='fixed':
            ################### Static mesh
            maxdepth=5
            density=0.2


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

            v=setup.iterativeSolve()
            lastNumEl=numEl
            setup.iteration+=1

            study.saveData(setup)#,baseName=str(setup.iteration))
        else:

            vdof=setup.getDoFs()
            v=setup.iterativeSolve(vGuess=vdof)

            # #Shortcut for single source
            # scale=ival/lastI
            # # print(scale)
            # v=setup.nodeVoltages*scale
            # setup.nodeVoltages=v

            # v=setup.solve()


        dt=time.monotonic()-t0

        lastI=ival

        study.newLogEntry(['Timestep','Meshnum'],[setup.currentTime, setup.meshnum])

        setup.stepLogs=[]
        numels.append(lastNumEl)

        print('%d percent done'%(int(100*tval/tmax)))






        errdict=xCell.misc.getErrorEstimates(setup)
        errdict['densities']=density
        errdict['depths']=maxdepth
        errdict['numels']=lastNumEl
        errdict['dt']=dt
        errdict['vMax']=max(np.abs(v))
        errdicts.append(errdict)


        if vids:
            err.addSimulationData(setup,append=True)
            img.addSimulationData(setup,append=True)

    lists=xCell.misc.transposeDicts(errdicts)
    pickle.dump(lists, open(studyPath+strat+'.p','wb'))
else:
    if vids:
        img.getStudyData()
        err.getStudyData()
    lists=pickle.load(open(studyPath+strat+'.p','rb'))


if post:
    ldep=pickle.load(open(studyPath+'depth.p','rb'))

    lk=pickle.load(open(studyPath+'k.p','rb'))

    lists=pickle.load(open(studyPath+'fixed'+'.p','rb'))
    f,axes=plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios':[5,1,1]})

    # f.suptitle('Adaptation: '+strat)



    # for mets in ['FVU','FVU2','powerError','int1','int3']:
    #     axes[0].plot(tvec,lists[mets],label=mets)

    # axes[0].set_ylabel('Error metric')


    # met='max'


    # met='int1'
    # lists['SSE']/np.sum(lists['SStot'])

    for dset,lbl in zip([lists,ldep,lk],['fixed','depth','density']):
        if dset['FVU']==[]:
            continue
        # met='nSSE'
        # dval=dset['SSE']/np.sum(dset['SSTot'])



        # # met='raw int1'
        # met='Relative intErr'
        # dval=np.abs(dset['intErr'])/np.abs(lists['intErr'])

        met='late-normalized int1'
        # intAna=np.abs(np.array(dset['intErr'])/np.abs(dset['int1']))

        dval=np.abs(dset['intErr'])/sum(dset['intAna'])


        # # met='SSE-seconds'
        # met = 'int1-seconds'
        # dval=dval*np.array(dset['dt'])


        # met='dt'
        # dval=np.array(dset[met])

        # dval=dset[met]*np.array(dset['dt'])
        axes[0].plot(tvec,dval,label=lbl)



        # axes[0].plot(tvec,np.abs(dset[met]),label=lbl)
        # axes[0].plot(tvec,dset[met]*abs(ivec)*dset['dt'],label=lbl)

    # axes[0].plot(tvec,lists[met],label='fixed')
    # axes[0].plot(tvec,ldep[met],label='depth')
    # axes[0].plot(tvec,lk[met],label='density')
    axes[0].set_ylabel(met)
    axes[0].set_yscale('log')


    axes[0].legend(loc='upper left')



    # a2.yaxis.set_major_formatter(xCell.Visualizers.eform('A'))
    # a2.grid(axis='y',color='C1')
    # axes[1].grid(axis='y',color='C0')

    #adjust vals

    def nicen(val):
        exp=np.floor(np.log10(val))
        return np.ceil(val/(10**exp))*10**exp



    # for aa in [a2,axes[1]]:
    #     aa.yaxis.set_ticks([0])



    axes[1].plot(tvec,lists['numels'])
    axes[1].plot(tvec,ldep['numels'])
    axes[1].plot(tvec,lk['numels'])


    axes[1].set_ylabel('Number of\nElements')
    axes[1].xaxis.set_major_formatter(xCell.Visualizers.eform('s'))

    axes[1].set_yscale('log')

    cvolt='C4'
    ccurr='C5'

    axes[2].plot(tvec,vvec,label='Voltage',color=cvolt)
    a2=plt.twinx(axes[2])

    vbnd=nicen(max(abs(vvec)))
    ibnd=nicen(imax)
    a2.set_ylim(-ibnd,ibnd)
    axes[2].set_ylim(-vbnd,vbnd)
    a2.plot(tvec,ivec,color=ccurr,label='Current')

    a2.set_ylabel('Current',color=ccurr)
    axes[2].set_ylabel('Voltage',color=cvolt)

    # study.savePlot(f, strat, '.png')
    # study.savePlot(f, strat, '.eps')



    plt.figure()
    A=plt.gca()

    # A.set_xlabel('Total time [seconds]')
    A.set_ylabel('Total error [int1]')
    # A.set_ylabel('Total error [FVU]')

    totT=[]
    totE=[]
    for data,lbl in zip([lists,ldep,lk],['fixed','depth','k']):
        if data['FVU']==[]:
            continue
        # E=sum(data['SSE'])/sum(data['SSTot'])
        E=sum(np.abs(data['intErr']))/sum(np.abs(data['intAna']))
        T=sum(data['dt'])
        totE.append(E)
        totT.append(T)
        A.scatter(T,E,label=lbl,s=10,marker='*')

        #
        # A.plot(data['dt'],np.abs(data['int1']), label=lbl)

    A.legend(loc='center')
    A.set_yscale('log')
    A.set_xscale('log')

    # bwidth=0.35
    # fbar=plt.figure()
    # bax1=fbar.add_subplot(111)
    # bax2=bax1.twinx()

    # bax1.bar(np.arange(3)-bwidth/2, totE, bwidth,color='C0')
    # bax2.bar(np.arange(3)+bwidth/2, totT, bwidth, color='C1')
    # plt.gca().set_xticks(np.arange(3), ['fixed','depth','k'])
    # # plt.legend()
    # bax1.set_ylabel('Error', color='C0')
    # bax2.set_ylabel('Sim time', color='C1')






if vids:
    ani=img.animateStudy('volt-'+strat,fps=30.)
    erAni=err.animateStudy('error-'+strat,fps=30.)


