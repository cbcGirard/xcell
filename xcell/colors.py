#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:24:37 2022

Color schemes for xcell

@author: benoit
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from os import path
import re

# Dark mode
MESH_ALPHA = 0.25
FAINT = (0xaf/255, 0xcf/255, 1., MESH_ALPHA)

colAr = [[0, 0, 1, 1],
         [.098, .137, .176, 0],
         [1, 0, 0, 1]]

CM_BIPOLAR = mpl.colors.LinearSegmentedColormap.from_list('bipolar',
                                                          np.array(colAr,
                                                                   dtype=float))

DARK = '#19232d'
HILITE = '#afcfff'
OFFWHITE = '#dcd4c7'
NULL = '#00000000'
WHITE = '#FFFFFF00'
ACCENT_DARK = '#990000'
ACCENT_LIGHT = '#FFCC00'

BASE = HILITE


plx = np.array(mpl.colormaps.get('plasma').colors)
lint = np.array(np.linspace(0, 1, num=plx.shape[0]), ndmin=2).transpose()
CM_MONO = mpl.colors.LinearSegmentedColormap.from_list('mono',
                                                       np.hstack((plx, lint)))



scopeColors=['#ffff00', '#00ffff', '#ff00ff', '#00ff00', '#ff0000', '#0000ff', '#ff8000', '#8000ff', '#ff0080', '#0080ff']
scopeColorsLite=['#ffcc00', '#17becf', '#e377c2', '#2ca02c', '#1c2a99', '#d62728', '#ff7f0e', '#9467bd', '#990000', '#7f7f7f']

styleScope = {
    'axes.prop_cycle': mpl.cycler('color', scopeColors)}

styleScope2 = {
    'axes.prop_cycle': mpl.cycler('color', scopeColorsLite)}


def makeStyleDict(fgColor,bgColor):
    """
    Generate dictionary of plotting preferences.

    Parameters
    ----------
    fgColor : color
        Color for gridlines, text, etc..
    bgColor : color
        Color for image background.

    Returns
    -------
    styleDict : dict
        Dict of matplotlib preferences (pass to mpl.style.use()).

    """
    bgCategories = ['axes.facecolor', 'figure.edgecolor', 'figure.facecolor',     'savefig.edgecolor', 'savefig.facecolor']

    fgCategories = ['axes.edgecolor', 'axes.labelcolor', 'boxplot.boxprops.color', 'boxplot.capprops.color', 'boxplot.flierprops.color', 'boxplot.flierprops.markeredgecolor', 'boxplot.whiskerprops.color', 'grid.color', 'lines.color', 'patch.edgecolor', 'text.color', 'xtick.color', 'ytick.color',]


    styleDict={
        'axes.grid': True,
        'figure.frameon': False,
        'figure.autolayout': True,
        'lines.markersize': 1.0,
        'legend.loc': 'upper right',
        'image.cmap': 'plasma',
        'image.aspect': 'equal',
        'image.origin': 'lower',

        'scatter.marker': '.',
        'path.simplify': True,

        'axes3d.grid': False,

        'grid.alpha': 0.5}

    for c in bgCategories:
        styleDict[c] = bgColor

    for c in fgCategories:
        styleDict[c] = fgColor


    return styleDict

def useDarkStyle():
    """
    Switch to dark-mode visualizations (suitable for screen).

    Returns
    -------
    None.

    """
    plt.style.use(makeStyleDict(fgColor=OFFWHITE, bgColor=DARK))
    plt.style.use(styleScope)

    global BASE
    BASE = HILITE


def useLightStyle():
    """
    Switch to light-mode visualizations (suitable for print).

    Returns
    -------
    None.

    """
    plt.style.use(makeStyleDict(fgColor=DARK, bgColor=WHITE))
    plt.style.use(styleScope2)

    global BASE
    BASE = DARK


def scoopCmap(baseCmap, fraction=0.1):
    """
    Fades colormap transparency as values approach 0.0.

    Parameters
    ----------
    baseCmap : colormap
        Colormap to modify.
    fraction : float, optional
        Fraction of colormap range to apply alpha fade. The default is 0.1.

    Returns
    -------
    newCmap : colormap
        Modified colormap.

    """
    col = np.array(baseCmap.colors)

    x = np.linspace(-1, 1, col.shape[0]).reshape((-1, 1))
    alpha = np.abs(x/fraction)
    alpha[alpha > 1.] = 1.

    newcol = np.hstack((col, alpha))

    newCmap = mpl.colors.LinearSegmentedColormap.from_list(
        baseCmap.name+'_mod', newcol)
    return newCmap




def recolorSVG(fname,toLight=True):
    fstr=open(fname)
    rawtxt=fstr.read()
    fstr.close()
    darkColors=[DARK, OFFWHITE, HILITE]
    liteColors=['#FFFFFF', DARK, ACCENT_DARK]

    darkColors.extend(scopeColors)
    liteColors.extend(scopeColorsLite)


    if toLight:
        origColors=darkColors
        newColors=liteColors
        tag='-lite'

    else:
        origColors=liteColors
        newColors=darkColors
        tag='-dark'

    for oldC,nuC in zip(origColors,newColors):
        regex=re.compile(re.escape(oldC), re.IGNORECASE)
        rawtxt=regex.sub(nuC,rawtxt)

    name,ext=path.splitext(fname)

    newfile=open(name+tag+ext,'w')

    newfile.write(rawtxt)
    newfile.close()
