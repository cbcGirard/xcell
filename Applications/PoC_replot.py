#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:45:31 2022

Better plotting of the stacked error plots for print

@author: benoit
"""

import xcell as xc
import Common as com
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


study, _ = com.makeSynthStudy('Quals/PoC/')
# study, _ = com.makeSynthStudy('Quals/formulations/')

# study, _ = com.makeSynthStudy('Quals/bigPOC/')

titles = [ 'Octree', 'Uniform',]

# vizU = study.load('ErrorGraph_uniform', '.adata')
# vizA = study.load('ErrorGraph_adaptive', '.adata')


# %%
frames = [2, 6, 14]

analytic = vizA.analytic
analyticR = vizA.analyticR

xc.colors.useLightStyle()

f = plt.figure(figsize=[6.5, 8])

# subs=f.subfigures(3,2)#,wspace=0.25,hspace=0.125)

# for nrow,row in enumerate(subs):
#     for v,sub in zip([vizU,vizA],row):

#         lastRow=nrow==2

#         v.fig=sub
#         v.setupFigure(labelX=lastRow, labelY=v==vizU)
#         v.prefs['showLegend']=(v==vizA and nrow==0)
#         v.analytic=analytic
#         v.analyticR=analyticR

#         v.getArtists(frames[nrow])

ratios = [5, 5, 3, 1]
hratio = ratios*3
hratio.pop()

subs = f.subplots(len(hratio), 2, gridspec_kw={'height_ratios': hratio})


l0span = xc.visualizers.ScaleRange()


def scootSubplot(axis, scootUp=True, fraction=0.05):
    # pos=axis.get_position().extents
    # #left, bottom, right, top
    # if scootUp:
    #     shift=np.array([0,-1.,0,0])
    # else:
    #     shift=np.array([0,0,0,1.])

    pos = axis.get_position().bounds
    if scootUp:
        shift = np.array([0, -1, 0, -1])
    else:
        shift = np.array([0, 1, 0, -1])

    axis.set_position(pos+fraction*shift)


for nrow in range(3):
    lastRow = nrow == 2
    for ncol in range(2):
        if ncol:
            v = vizA
            tstr = 'Octree'
        else:
            v = vizU
            tstr = 'Uniform'

        if nrow == 0:
            subs[nrow, ncol].set_title(tstr)

        v.axes = subs[4*nrow:4*nrow+3, ncol]
        v.setupFigure(labelX=lastRow,
                      labelY=v == vizU,
                      # labelY=False,
                      newAxes=False)

        v.prefs['showLegend'] = False
        v.analytic = analytic
        v.analyticR = analyticR

        v.getArtists(frames[nrow])

        l0span.update(v.dataSets[frames[nrow]]['elemL'])

subs[0, 1].legend()
[ax.set_ylim(l0span.min, l0span.max) for row in subs[2::4] for ax in row]
[ax.axis('off') for row in subs[3::4] for ax in row]
# for ii, row in enumerate(subs):
#     if ii%3==1:
#         continue
#     for sub in row:
#         scootSubplot(sub,ii%3)


f.align_labels()


# %% Error vs. numel and time
xc.colors.useLightStyle()

logfile = study.studyPath+'/log.csv'
df, cats = study.loadLogfile()

# xaxes=['Number of elements','Total time [Wall]']
xaxes=['Number of elements','l0min']
group = 'Mesh type'

# group='Element type'
# xaxes=['adaptive','FEM','Face']
l0string=r'Smallest $\ell_0$ [m]'


def logfloor(val):
    return 10**np.floor(np.log10(val))
def logceil(val):
    return 10**np.ceil(np.log10(val))

def loground(axis):
    # xl=axis.get_xlim()
    # yl=axis.get_ylim()

    lims = [[logfloor(aa[0]), logceil(aa[1])] for aa in axis.dataLim.get_points().transpose()]#[xl, yl]]
    axis.set_xlim(lims[0])
    axis.set_ylim(lims[1])



with mpl.rc_context({'lines.markersize': 5,  'lines.linewidth': 2}):
    f, axes = plt.subplots(2, 1, sharey=True, figsize=[6.5, 8])
    f2,a2=plt.subplots(1,1,figsize=[6.5,3])



    for ax, xcat in zip(axes, xaxes):
        xc.visualizers.groupedScatter(
            logfile, xcat, ycat='Error', groupcat=group, ax=ax)
        # ax.set_ylim(bottom=logfloor(ax.get_ylim()[0]))

    xc.visualizers.groupedScatter(logfile, ycat='Total time [Wall]', xcat='l0min', groupcat=group, ax=a2, df=df[2:])

[loground(a) for a in axes]


axes[1].invert_xaxis()
axes[0].legend(labels=titles)
axes[1].set_xlabel(l0string)
a2.legend(loc='lower right', labels=titles)
loground(a2)
a2.set_xlabel(l0string)

a2.invert_xaxis()

study.savePlot(f, 'Error_composite')
study.savePlot(f2, 'PerformanceSummary')

# %% Computation time vs element size
# xaxes=['Number of elements','Error']
xaxes=['Number of elements','l0min']


isoct=df['Mesh type']=='adaptive'
gentimes=df['Make elements [Wall]'][isoct].to_numpy()
newtime=gentimes-np.roll(gentimes,1)
newtime[:2]=0

with mpl.rc_context({'figure.figsize':[6.5,3]}):
    f, axes = plt.subplots(1,2,sharex=True)
    # f.subplots_adjust(bottom=0.3)

    # f, axes = plt.subplots(2, 1,sharex=True)#, gridspec_kw={'height_ratios':[2,4]})
    # axes[1,0].invert_xaxis()

    # for arow,xcat in zip(axes,xaxes):

    #     arow[1].sharex(arow[0])
    #     arow[1].sharey(arow[0])
    #     for ax, mtype, title in zip(arow, ['uniform', 'adaptive'], ['Uniform', 'Octree']):
    for ax, mtype, title in zip(axes, ['adaptive', 'uniform'], titles):
            # if xcat==xaxes[1]:
            #     xc.visualizers.groupedScatter(fname=logfile, xcat=xcat, ycat='Total time [Wall]', groupcat='Mesh type', ax=ax)
            #     ax.set_yscale('log')
            #     ax.set_xscale('log')
            #     # pass
            # else:
            ax.set_title(title)
            xc.visualizers.importAndPlotTimes(
            fname=logfile, timeType='Wall', ax=ax, xCat='Number of elements', onlyCat='Mesh type', onlyVal=mtype)

axes[1].set_ylabel('')
f.align_labels()
axes[0].set_xlim(left=0)



# axes[1].stackplot(df['Number of elements'][isoct], newtime, baseline='zero',color=0.5*np.ones(4))
# xc.visualizers.outsideLegend(where='bottom',ncol=3)
# plt.tight_layout()

study.savePlot(f, 'PerformanceStack')

# %% Computation time vs element size plus parallelism
# xaxes=['Number of elements','Error']
xaxes=['Number of elements','l0min']


isoct=df['Mesh type']=='adaptive'
gentimes=df['Make elements [Wall]'][isoct].to_numpy()
newtime=gentimes-np.roll(gentimes,1)
newtime[:2]=0

with mpl.rc_context({'figure.figsize':[6.5,5]}):

    f, axes = plt.subplots(2, 2, sharex=True)#, gridspec_kw={'height_ratios':[2,4]})
    # axes[1,0].invert_xaxis()

    for arow,xcat in zip(axes,xaxes):

        arow[1].sharey(arow[0])
        for ax, mtype, title in zip(arow, ['uniform', 'adaptive'], titles):

            if xcat==xaxes[1]:
                xc.visualizers.importAndPlotTimes(
                fname=logfile, timeType='Ratio', ax=ax, xCat='Number of elements', onlyCat='Mesh type', onlyVal=mtype)

                ax.set_yscale('linear')
                # ax.set_xscale('log')
                # pass
            else:
                ax.set_title(title)
                xc.visualizers.importAndPlotTimes(
            fname=logfile, timeType='Wall', ax=ax, xCat='Number of elements', onlyCat='Mesh type', onlyVal=mtype)
                ax.set_xlabel('')

            # if mtype='adaptive':
        arow[1].set_ylabel('')



f.align_labels()
axes[0,0].set_xlim(left=0)

# axes[1].stackplot(df['Number of elements'][isoct], newtime, baseline='zero',color=0.5*np.ones(4))




#%% Image slice redux
# v=pickle.load(open('/home/benoit/smb4k/ResearchData/Results/Quals/PoC/SliceSet_adaptive.adata','rb'))

with plt.rc_context({'figure.figsize':[7,6.5],
                     'figure.dpi':144}):
    # v2=v.copy(overridePrefs={'logScale':True})
    study,_=com.makeSynthStudy('Quals/bigPOC')
    v=xc.visualizers.SliceSet(None, study)
    v.getStudyData()
    v.animateStudy('ImageTst')

    # v2.prefs['logScale']=True
    # v2.dataScales['vbounds'].knee=1e-3
    # v2.dataScales['errbounds'].knee=0.1
    # v2.animateStudy('Image')