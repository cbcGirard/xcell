#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:00:57 2022

@author: benoit
"""

import xcell as xc
import numpy as np
import Common as com
from neuron import h#, gui
import neuron.units as nUnit

import matplotlib.pyplot as plt


domX=250e-6

stepViz=False


pairStim=False
#https://doi.org/10.1109/TBME.2018.2791860
#48um diameter +24um insulation; 1ms, 50-650uA

Imag=-150e-6
tstop=20.
tpulse=1.
tstart=5.
rElec=24e-6

cellY=250

study,_=com.makeSynthStudy('NEURON/Stim/Mono',xmax=domX,)


if stepViz:
    viz=xc.visualizers.SliceSet(None,study,
                                prefs={
                                    'showError':False,
                                    'showInsets':False,
                                    'relativeError':False,
                                    'logScale':True,
                                    'showNodes':True,
                                    'fullInterp':True})
    #zoom in
    # resetBounds(viz.axes[0], 2e-4)

def resetBounds(ax,xmax,xmin=None):
    if xmin is None:
        xmin=-xmax

    tix=np.linspace(xmin,xmax,3)

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(xmin,xmax)
    ax.set_xticks(tix)
    ax.set_yticks(tix)



elec0=np.zeros(3)
elec0[1]=cellY*1e-6
elecA=elec0.copy()
elecB=elec0.copy()

if pairStim:
    elecA[0]=2*rElec
    elecB[0]=-2*rElec
    wire2=xc.geometry.Disk(elecB,
                        rElec,
                        np.array([0,0,1]))

    if stepViz:
        xc.visualizers.showSourceBoundary(viz.axes, rElec,
                                      srcCenter=elecB[:-1])

if stepViz:
    xc.visualizers.showSourceBoundary(viz.axes, rElec,
                                  srcCenter=elecA[:-1])



wire1=xc.geometry.Disk(elecA,
                    rElec,
                    np.array([0,0,1],dtype=float))



def runStim(depth):
    h.load_file('stdrun.hoc')
    # h.CVode().use_fast_imem(1)



    study,setup=com.makeSynthStudy('NEURON/Stim',xmax=domX,)
    setup.currentSources[0].geometry=wire1
    setup.currentSources[0].value=Imag

    if pairStim:
        setup.addCurrentSource(-Imag, elecB)
        setup.currentSources[1].geometry=wire2

    # depth=8


    metrics=[]
    for src in setup.currentSources:
        metrics.append(xc.makeExplicitLinearMetric(maxdepth=depth,
                                                   meshdensity=0.2,
                                                   origin=src.geometry.center))
    # metric= xc.makeExplicitLinearMetric(maxdepth=depth,
    #                             meshdensity=0.2)

    setup.makeAdaptiveGrid(metrics, depth)

    setup.finalizeMesh()
    setup.setBoundaryNodes()
    v=setup.iterativeSolve(tol=1e-9)
    numel=len(setup.mesh.elements)
    numElec=sum(setup.nodeRoleTable==2)

    def testThresh(k):

        # ring=com.Ring(N=1,stim_delay=0)
        cell=com.BallAndStick(1, -50., 0., 0., 0.)
        cell.dend.nseg=15
        h.define_shape()

        h.nlayer_extracellular(1)
        cellcoords=[]
        inds=[]
        for nsec,sec in enumerate(h.allsec()):
            sec.insert('extracellular')

            ptN=sec.n3d()

            for ii in range(ptN):
                cellcoords.append([sec.x3d(ii), sec.y3d(ii), sec.z3d(ii)])
                inds.append([[nsec,ii]])

        #optional visualization
        if stepViz:
            viz.addsimulationdata(setup,append=True)
            xc.nrnUtil.showcellgeo(viz.axes[0])





        vext=setup.interpolateAt(np.array(cellcoords)/nUnit.m)



        tstim,vstim =xc.nrnUtil.makeBiphasicPulse(k, tstart, tpulse)

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


        h.finitialize(-65*nUnit.mV)

        h.continuerun(tstop)

        memVals=cell.soma_v.as_numpy()
        t=tvec.as_numpy()

        nElec=sum(setup.nodeRoleTable==2)

        spiked=np.any(memVals>0)

        del cell

        for sec in h.allsec():
            h.delete_section(sec=sec)

        # for vec in h.allobjects('Vector'):
        #     vec.play_remove()
        tvec.play_remove()
        [v.play_remove() for v in vvecs]
        [v.play_remove() for v in vmems]



        return spiked


    def findThresh():
        mn=1e-3
        mx=1e3

        while (mx-mn)>1e-6:
            md=0.5*(mn+mx)
            print(md)
            spike=testThresh(md)

            if spike:
                mx=md
            else:
                mn=md

        return md

    thresh=findThresh()

    print(Imag*thresh)

    return thresh, numel, nElec

threshSet=[]
nElec=[]
nTot=[]
f,axes=plt.subplots(2,sharex=True,
                    gridspec_kw={'height_ratios':[4,1]})


# TODO: fix problems with appending
for d in range(5,14):
    t,n,nElec=runStim(d)
    # axes[0].plot(t,v,label=str(nElec)+'/'+str(n))


    # vset.append(v)
    # numels.append(n)




# axes[1].plot(t,np.array(vvecs).transpose())
axes[1].step([0,tstart,tpulse+tstart,tstart+2*tpulse,tstop],
             1e6*np.array([0.,0.,Imag, -Imag,0]))
axes[0].set_ylabel('Membrane voltage [mV]')

axes[1].set_xlabel('Time [ms]')
axes[1].set_ylabel(r'Stimulus [$\mu$A]')

axes[0].legend()

ani=viz.animateStudy('dualTest')

# del cell

for sec in h.allsec():
    h.delete_section(sec=sec)

# h.quit()
