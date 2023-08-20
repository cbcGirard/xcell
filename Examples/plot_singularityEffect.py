#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Singularity
==========================

Get a detailed sweep of the behavior as element l0 approaches the source's radius

"""

import xcell as xc
import numpy as np
import Common_nongallery
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib as mpl

fname = 'Singularity_sweep'
# fname="Singularity"
animate = False

rElec = 1e-6
x0max = rElec*200
study0, _ = Common_nongallery.makeSynthStudy(fname,
                                  rElec=rElec,
                                  xmax=x0max)

elRs = 1e-6*np.array([1, 5])
x0s = np.array([50, 500])
ks = [0.2, 0.4]

# elRs=1e-6*np.array([1])
# x0s=np.array([200])
# ks=[0.2]
# with mpl.rc_context({'lines.markersize': 2,
#                      'lines.marker': 'o',
#                      'lines.linewidth': 1,
#                      'font.size': 10,
#                     'figure.figsize': [3.25, 5]}):
with mpl.rc_context({'lines.markersize': 2.5,
                     'lines.marker': 'o',
                     'lines.linewidth': 1.5,
                     'font.size': 10,
                    'figure.figsize': [6.5, 5.5]}):

    f2, axes = plt.subplots(3, sharex=True, gridspec_kw={
                            'height_ratios': [4, 4, 2]})
    [ax.grid(True) for ax in axes]
    ax = axes[0]
    ax2 = axes[1]
    a3 = axes[2]

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_inverted(True)
    ax2.set_ylabel('Error at source [V]')
    a3.set_ylabel('Points in source')
    # a3.set_xlabel('Closest node to origin [m]')

    a3.set_xlabel(r'Ratio of $\ell_0$ to source radius')
    a3.set_yscale('log')

    ax.set_ylabel('Total error')

    for xx in x0s:
        for rElec in elRs:
            for k in ks:
                x0max = xx*rElec
                condStr = "r%dx%dk%d" % (int(rElec*1e6), xx, int(10*k))
                fbase = study0.study_path+'/'+condStr

                if animate:
                    plotter = xc.visualizers.ErrorGraph(plt.figure(),
                                                            study0)
                    plotter.axes[0].xaxis.set_inverted(True)

                def boundary_function(coord):
                    r = np.linalg.norm(coord)
                    val = rElec/(r*np.pi*4)
                    return val

                etots = []
                esrc = []
                nInSrc = []
                rclosest = []
                rrel = []

                logmin = int(np.floor(np.log10(rElec/2)))-1
                logmax = logmin+2

                for min_l0 in np.logspace(logmax, logmin):
                    max_depth = int(np.floor(np.log2(x0max/min_l0)))
                    xmax = min_l0*2**max_depth
                    study, setup = Common_nongallery.makeSynthStudy(fname,
                                                            xmax=xmax,
                                                            rElec=rElec)


                    setup.quick_adaptive_grid(max_depth = max_depth, coefficent=k)
                    setup.set_boundary_nodes(boundary_function)

                    v = setup.solve()

                    setup.getMemUsage(True)
                    setup.print_total_time()

                    if animate:
                        plotter.add_simulation_data(setup)
                    emetric, evec, _, sortr, _ = setup.calculate_errors()
                    etots.append(emetric)
                    esrc.append(evec[sortr][0])
                    nInSrc.append(sum(setup.node_role_table == 2))

                    r = np.linalg.norm(setup.mesh.node_coords, axis=1)

                    rclose = min(r[r != 0])

                    rclosest.append(rclose)
                    rrel.append(rclose/rElec)

                if animate:
                    ani = plotter.animate_study(fbase, fps=10)

                pdata = {
                    'condStr': condStr,
                    'rclosest': rclosest,
                    'rrel': rrel,
                    'etots': etots,
                    'esrc': esrc,
                    'nInSrc': nInSrc}

                pickle.dump(pdata, open(fbase+'.pdata', 'wb'))
                

                # totcol='tab:orange'
                # pkcol='tab:red'
                totcol = 'k'
                pkcol = 'k'

                rplot = np.array(rrel)

                sortr = np.argsort(rplot)
                ax.plot(rplot[sortr], np.array(etots)[sortr], label=condStr)

                ax2.plot(rplot[sortr], np.array(esrc)[sortr])

                a3.plot(rplot[sortr], np.array(nInSrc)[sortr])



xc.util.loground(ax, which='y')
f2.align_labels()

for a in axes:
    y0, y1 = a.get_ylim()
    a.vlines(np.pi, y0, y1, linestyle='dashed', color=xc.colors.BASE)
    a.text(np.pi,y0,r'$\uparrow \ell_0 = \pi r_0$')