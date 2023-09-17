#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the breakdown of step times within the hippocampus simulation
"""

import xcell as xc
import numpy as np
import matplotlib.pyplot as plt
import os

fullwidth = True
if fullwidth:
    widthtag = '-fullwidth'
    figsize = [6.5, 4]
else:
    widthtag = ''
    figsize = [3.25, 2]


def getTimes(foldername):

    study = xc.Study(os.path.join(os.getcwd(), foldername),
                        bounding_box=np.ones(6))

    df, cats = study.load_logfile()
    order = np.argsort(df['Number of elements'])

    plotcats = [s for s in cats if s.find('Wall') > 0]
    plotcats.pop()  # skip total time category

    errthreshold = 1e2  # redact bad values

    labels = []
    data = []
    for s in plotcats:
        strim, _ = s.split(' [')
        labels.append(strim)
        vals = df[s][order]
        vals[vals > errthreshold] = 0
        data.append(vals)

    return df['Number of elements'][order], np.array(data), labels


xc.colors.use_light_style()

with plt.rc_context({
        'figure.figsize': figsize}):

    fig, ax = plt.subplots()
    xlocs = np.arange(2)
    names = ['Static', 'Dynamic']

    for f, name, ii in zip(['full-d16x1', 'adaptive-full-d16x1'],
                           names, xlocs):
        x, dat, labels = getTimes(f)
        summary = np.nansum(dat, axis=1, keepdims=True)
        xspan = ii + 0.3*(-1)**np.arange(2)
        ax.stackplot(xspan, np.tile(summary, 2), labels=labels)
        if ii == 0:
            plt.legend()

        ax.set_prop_cycle(None)

    ax.set_xticks(xlocs)
    ax.set_xticklabels(names)
    ax.grid(False)
    ax.grid(True, axis='y')
    ax.set_ylabel('Total simulation time [s]')
    # plt.stackplot(df['Number of elements'][order], np.array(data),
    #               labels=labels)
    # plt.legend()

    if ii == 1:
        study = xc.Study(os.path.join(os.getcwd(), f),
                            bounding_box=np.ones(6))
        study.save_plot(fig, 'stepBreakdown'+widthtag)
