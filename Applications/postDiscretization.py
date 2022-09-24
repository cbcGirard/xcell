#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:19:43 2022

@author: benoit
"""

import xcell
import pickle
import matplotlib.pyplot as plt
import numpy as np
import Common as com
import os


class Composite(xcell.visualizers.FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        super().__init__(fig, study,prefs)

    def setupFigure(self, resetBounds = False):
        rAnalytic = 0.5*np.logspace(-6,-3)
        vAnalytic = 1e-6/rAnalytic
        vAnalytic[rAnalytic<1e-6]=1.

        ax = plt.gca()
        ax.semilogx(rAnalytic,vAnalytic, color = xcell.colors.BASE, label = 'Analytic')


        self.axes.append(ax)

        xcell.visualizers.engineerTicks(ax,'m','V')

        ax.set_xlim(rAnalytic[0], rAnalytic[-1])
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
            ax.legend(loc='upper right')

        return artists



# folder = '/home/benoit/smb4k/ResearchData/Results/Quals/formulations/'
study, _ = com.makeSynthStudy('Quals/formulations')
folder = study.studyPath
xcell.colors.useDarkStyle()


formulations = ['Admittance', 'FEM', 'Face']
titles = ['Admittance', 'FEM', 'Dual']


newGraph = Composite(None, study)
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

    # graphdata=xcell.misc.transposeDicts(graph.dataSets)

    # finalData['v'+title]=graphdata['simV']
    # finalData['r'+title]=graphdata['simR']




newGraph.animateStudy('Composite')#,artists = artists)

xcell.colors.useLightStyle()
liteGraph = newGraph.copy()

liteGraph.animateStudy('Composite-lite')