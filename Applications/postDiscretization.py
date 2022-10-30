#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:19:43 2022

Post-processing of alternate discretizations (FEM, mesh dual,...)

@author: benoit
"""

import xcell as xc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import Common as com
import os
import matplotlib as mpl

class Composite(xc.visualizers.FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        super().__init__(fig, study,prefs)

    def showAnalytic(self, ax=None):
        rAnalytic = 0.5*np.logspace(-6,-3)
        vAnalytic = 1e-6/rAnalytic
        vAnalytic[rAnalytic<1e-6]=1.

        if ax is None:
            ax = plt.gca()
        ax.semilogx(rAnalytic,vAnalytic, color = xc.colors.BASE, label = 'Analytic')
        ax.set_xlim(rAnalytic[0], rAnalytic[-1])

        return ax


    def setupFigure(self, resetBounds = False):

        ax=self.showAnalytic()
        self.axes.append(ax)

        xc.visualizers.engineerTicks(ax,'m','V')

        ax.set_ylim(bottom = 1e-3, top = 1e1)
        ax.set_ylabel('Voltage')
        ax.set_xlabel('Distance')


    def getArtists(self, setnumber, data=None):
        ax = self.axes[0]


        if data is None:
            data = self.dataSets[setnumber]


        titles = ['Admittance', 'FEM', 'Dual']

        artists = []
        for n,title in enumerate(titles):
            artists.extend(ax.loglog(data['r'+title],
                                       data['v'+title],
                                       color = 'C'+str(n),
                                       label = title))



        if setnumber==0:
            artists.append(ax.legend(loc='upper right'))

        return artists



# folder = 'Quals/formulations'
folder = 'Quals/fixedDisc'


study, _ = com.makeSynthStudy('folder')
folder = study.studyPath
xc.colors.useDarkStyle()


formulations = ['Admittance', 'FEM', 'Face']
titles = ['Admittance', 'Trilinear FEM', 'Mesh dual']


#%%

f=plt.figure(figsize=[6.5*.6, 8/3])

newGraph = Composite(f, study)
ax = newGraph.axes[0]

for form,title in zip(formulations, titles):
    graph = pickle.load(open(os.path.join(folder, 'ErrorGraph_'+form+'.adata'),'rb'))

    for ii,dat in enumerate(graph.dataSets):
        dnew = {'v'+title:dat['simV'],
                'r'+title:dat['simR']}
        if ii>=len(newGraph.dataSets):
            newGraph.dataSets.append(dnew)
        else:
            newGraph.dataSets[ii].update(dnew)

    # graphdata=xc.misc.transposeDicts(graph.dataSets)

    # finalData['v'+title]=graphdata['simV']
    # finalData['r'+title]=graphdata['simR']




# newGraph.animateStudy('Composite')#,artists = artists)


#%%
xc.colors.useLightStyle()
frames = [2, 6, 10]

f,axes = plt.subplots(3,1,figsize=[6.5, 8], sharex=True)
liteGraph = newGraph.copy(newFigure=f)


for ax, frame in zip(axes,frames):
    liteGraph.axes=[ax]
    liteGraph.showAnalytic(ax)

    # liteGraph.getArtists(frame)
    for t in titles:
        ax.loglog(liteGraph.dataSets[frame]['r'+t],
                  liteGraph.dataSets[frame]['v'+t],
                  label = t)

    ax.set_ylabel('Potential [V]')
    ax.set_ylim(bottom=1e-3)

    if frame==frames[0]:
        ax.legend()

    if frame==frames[-1]:
        ax.set_xlabel('Distance [m]')


study.savePlot(f, 'frames-lite')


#%% Summary graphs
xc.colors.useLightStyle()

logfile = study.studyPath+'/log.csv'
df, cats = study.loadLogfile()

# xaxes=['Number of elements','Total time [Wall]']
xaxes=['Number of elements']
# group = 'Mesh type'

group='Element type'
# xaxes=['adaptive','FEM','Face']
l0string=r'Smallest $\ell_0$ [m]'


def logfloor(val):
    return 10**np.floor(np.log10(val))
def logceil(val):
    return 10**np.ceil(np.log10(val))

def loground(axis,which='both'):
    # xl=axis.get_xlim()
    # yl=axis.get_ylim()

    lims = [[logfloor(aa[0]), logceil(aa[1])] for aa in axis.dataLim.get_points().transpose()]#[xl, yl]]
    if which=='x' or which=='both':
        axis.set_xlim(lims[0])
    if which=='y' or which=='both':
        axis.set_ylim(lims[1])



with mpl.rc_context({'lines.markersize': 5,  'lines.linewidth': 2}):
    f, axes = plt.subplots(1, 1, sharey=True, figsize=[6.5, 4])



    xc.visualizers.groupedScatter(
            logfile, xcat='Number of elements', ycat='Error', groupcat=group, ax=axes)
        # ax.set_ylim(bottom=logfloor(ax.get_ylim()[0]))


# loground(axes)
axes.set_yscale('linear')
loground(axes,which='x'd)

# axes.invert_xaxis()
axes.legend(labels=titles)
# axes.set_xlabel(l0string)

study.savePlot(f, 'Error_composite')
# study.savePlot(f2, 'PerformanceSummary')



# %% Computation time vs element size
# xaxes=['Number of elements','l0min']
xaxes=['Number of elements','Error']


isoct=df['Mesh type']=='adaptive'
gentimes=df['Make elements [Wall]'][isoct].to_numpy()
newtime=gentimes-np.roll(gentimes,1)
newtime[:2]=0

with mpl.rc_context({'figure.figsize':[6.5,3]}):
    f, axes = plt.subplots(1,3,sharex=True,sharey=True)
    # f.subplots_adjust(bottom=0.3)

    # f, axes = plt.subplots(2, 1,sharex=True)#, gridspec_kw={'height_ratios':[2,4]})
    # axes[1,0].invert_xaxis()

    # for arow,xcat in zip(axes,xaxes):

    #     arow[1].sharex(arow[0])
    #     arow[1].sharey(arow[0])
    #     for ax, mtype, title in zip(arow, ['uniform', 'adaptive'], ['Uniform', 'Octree']):
    for ax, mtype, title in zip(axes, ['Admittance', 'FEM', 'Face'], titles):
            # if xcat==xaxes[1]:
            #     xc.visualizers.groupedScatter(fname=logfile, xcat=xcat, ycat='Total time [Wall]', groupcat='Mesh type', ax=ax)
            #     ax.set_yscale('log')
            #     ax.set_xscale('log')
            #     # pass
            # else:
            ax.set_title(title)
            xc.visualizers.importAndPlotTimes(
            fname=logfile, timeType='Wall', ax=ax, xCat='Number of elements', onlyCat='Element type', onlyVal=mtype)

[a.set_ylabel('') for a in axes[1:]]
f.align_labels()
axes[0].set_xlim(left=0)



# axes[1].stackplot(df['Number of elements'][isoct], newtime, baseline='zero',color=0.5*np.ones(4))
# xc.visualizers.outsideLegend(where='bottom',ncol=3)
# plt.tight_layout()

study.savePlot(f, 'PerformanceStack')
