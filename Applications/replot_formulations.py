#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ad-hoc recreations of formulation data: closeup of errors vs. distance from source at gen 11
"""

import xcell as xc
import Common as com
import pickle
import os
import matplotlib.pyplot as plt


folder = 'Quals/formulations'

nthGen = 11


study, _ = com.makeSynthStudy(folder)
folder = study.studyPath
xc.colors.useLightStyle()


formulations = ['Admittance', 'FEM', 'Face']
titles = ['Admittance', 'Trilinear FEM', 'Mesh dual']

rs = []
vs = []

for form, title in zip(formulations, titles):
    graph = pickle.load(
        open(os.path.join(folder, 'ErrorGraph_'+form+'.adata'), 'rb'))

    dat = graph.dataSets[nthGen]

    rs.append(dat['simR'])
    vs.append(dat['simV'])

    l0 = dat['elemL']
    elR = dat['elemR']

    dnew = {'v'+title: dat['simV'],
            'r'+title: dat['simR']}
    dat.update(dnew)

# dat=newGraph.dataSets[nthGen]
# rs=[k for k in dat.keys() if k[0]=='r']
# vs=[k for k in dat.keys() if k[0]=='v']
# ls=[k[1:] for k in vs]


# sim=study.loadData('sim11')
# l0 = graph.dataSets[11]['elemL']
# elR = graph.dataSets[11]['elemR']


# %%
with plt.rc_context({
    'lines.markersize': 0.5,
    'lines.linewidth': 1,
    'figure.figsize': [3.25, 3],
    'font.size': 10,
    'legend.fontsize': 10,
        'axes.prop_cycle': plt.rcParams['axes.prop_cycle'][:4] + plt.cycler('linestyle', ['-', '--', ':', '-.'])}):
    f, axes = plt.subplots(2, 1, sharex=True,
                           gridspec_kw={
                               'height_ratios': [5, 1]
                           })

    axes[0].set_xscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xlim(5e-7, max(rs[0]))

    axes[0].set_ylabel('Absolute error [V]')
    axes[1].set_xlabel('Distance from source')
    axes[1].set_ylabel(r'$\ell_0$ [m]')

    axes[1].scatter(elR, l0, c=xc.colors.BASE, marker='.')

    # xc.visualizers.engineerTicks(axes[0], 'm', 'V')
    xc.visualizers.engineerTicks(axes[1], 'm', None)

    for r, v, l in zip(rs, vs, titles):

        an = 1e-6/r
        an[r <= 1e-6] = 1.0
        er = v-an
        axes[0].plot(r, er, label=l)

    axes[0].legend()
    f.align_ylabels()


study.savePlot(f, 'FormulationErrorsDetail')
