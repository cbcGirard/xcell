#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:06:06 2022

Postprocessing of dynamic adaptation tests

@author: benoit
"""

import xcell
import Common
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import xcell.nrnutil as nUtil

# studyPath='Quals/polyCell'
# nRing=5
# ring = Common.Ring(N=nRing, stim_delay=0, dendSegs=101, r=175)
# tstop = 40
# tPerFrame = 5
# barRatio=[9,1]

studyPath='Quals/monoCell'
nRing=0
tstop = 12
tPerFrame=2
barRatio = [4, 1]

study,_=Common.makeSynthStudy(studyPath)
folderstem=study.studyPath


dmin=4
dmin=8


comDict=study.load('commonData','.xstudy')
tv=comDict['tv']
I=comDict['I']
rads=comDict['rads']
coords=comDict['coords']
isSphere=comDict['isSphere']
polygons=comDict['polygons']
# %%

def getdata(strat):
    fname = os.path.join(folderstem, strat, strat+'.pcr')
    data = pickle.load(open(fname, 'rb'))
    return data

labels=[l for l in os.listdir(folderstem) if os.path.isdir(os.path.join(folderstem,l))]
labels.sort()
labels=[labels[n] for n in [3,0,2,1]]
data=[getdata(l) for l in labels]


for fmt in ['.png','.svg','.eps']:
    for lite in ['','-lite']:
        if lite=='':
            xcell.colors.useDarkStyle()
        else:
            xcell.colors.useLightStyle()

        f, ax = plt.subplots(3, gridspec_kw={'height_ratios': [5, 5, 2]})
        ax[2].plot(tv, I)

        for d, l in zip(data, labels):
            # ax[0].semilogy(tv, np.abs(d['volErr']), label=d['depths'])
            ax[0].plot(tv,  np.abs(d['intErr']), label=d['depths'])
            ax[1].plot(tv[1:], d['dt'][1:])

        ax[0].legend()
        ax[0].set_ylabel('Error')
        ax[1].set_ylabel('Wall time')
        ax[2].set_ylabel('Activity')
        ax[2].set_xlabel('Simulated time')

        xcell.visualizers.engineerTicks(ax[2], xunit='s')
        [a.set_xticks([]) for a in ax[:-1]]



        f2,ax=plt.subplots(1,1)
        aright=ax.twinx()

        # [a.set_yscale('log') for a in [ax,aright]]

        ttime=[sum(l['dt']) for l in data]
        terr=[sum(np.abs(l['intErr'])) for l in data]
        # terr=[sum(np.abs(l['intErr']))/sum(np.abs(l['intAna'])) for l in data]
        # terr=[sum(np.abs(l['SSE']))/sum(np.abs(l['SSTot'])) for l in data]

        categories=[d['depths'] for d in data]
        barpos=np.arange(len(categories))
        barw=0.25

        tart=ax.bar(barpos-barw/2,height=ttime, width=barw, color='C0', label='Time')
        ax.set_ylabel('Total simulation time')

        eart=aright.bar(barpos+barw/2, height=terr, width= barw, color='C1', label='Error')
        aright.set_ylabel('Total simulation error')

        ax.set_xticks(barpos)
        ax.set_xticklabels(categories)

        ax.legend(handles=[tart, eart])

        axes=[ax, aright]
        ntick=0

        nticks=[len(a.get_yticks()) for a in axes]
        ntick=max(nticks)
        for a in axes:
            dtick=a.get_yticks()[1]
            a.set_yticks(np.arange(ntick)*dtick)


        study.savePlot(f2, os.path.join(
            folderstem, 'ring%dsummary%s' % (nRing, lite)), ext=fmt)

        study.savePlot(f, os.path.join(folderstem, 'ring%dcomparison%s' % (nRing, lite)), ext=fmt)



# #%% Semi- manual relabling
# for l in labels:
#     d=getdata(l)
#     if l[:5]=='fixed':
#         if l[5:]=='Min':
#             dstr=
#         else:
#             dstr=dmax
#         s='Fixed depth %d'%dstr
#     else:
#         if len(l)>5:
#             dmax=12
#         else:
#             dmax=8
#         s='Dynamic depth [%d:%d]'%(dmin,dmax)
#     d['depths']=s
#     pickle.dump(d,open(os.path.join(folderstem,l,l+'.pcr'), 'wb'))

#%% voltage plots
an=pickle.load(open(os.path.join(folderstem,'depth','volt-depth.adata'),'rb'))
fname='volt-depth'

xcell.colors.useLightStyle()

alite=an.copy({'colorbar':False,
               'barRatio':barRatio})

figw=0.3*6.5
alite.fig.set_figwidth(figw)
alite.fig.set_figheight(1.2*figw)
alite.axes[0].set_xticks([])
alite.axes[0].set_yticks([])
alite.tbar.axes[0].set_xlim(right=tstop/1000)

nUtil.showCellGeo(alite.axes[0])

#get closest frames to 5ms intervals
frameNs=[int(f*len(alite.dataSets)/tstop) for f in np.arange(0,tstop, tPerFrame)]
artists=[alite.getArtists(ii) for ii in frameNs]
alite.animateStudy(fname+'-lite', fps=30, artists=artists, vectorFrames=np.arange(len(frameNs)), unitStr='V')