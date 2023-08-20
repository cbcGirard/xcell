#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Color schemes for xcell."""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap as LinCM
import numpy as np

from os import path, environ
import re
import cmasher as cm

import pyvista as pv
from pyvista import themes

# TODO: autodetect remote execution?
# pyvista startup for remote server
if 'DISPLAY' not in environ:
    pv.start_xvfb()

pv.set_jupyter_backend('trame')
pv.global_theme.trame.server_proxy_enabled = True
pv.global_theme.trame.server_proxy_prefix = '/proxy/'

# Dark mode
#: Opacity of edges
MESH_ALPHA = 0.25
#: Color of edges
FAINT = (0xaf/255, 0xcf/255, 1., MESH_ALPHA)


def scoop_cmap(baseCmap, fraction=0.1):
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

    newCmap = LinCM.from_list(
        baseCmap.name+'_mod', newcol)
    return newCmap


#: Default colormap for bipolar data
CM_BIPOLAR = scoop_cmap(cm.guppy_r, 0.5)


# color palette
DARK = '#19232d'
HILITE = '#afcfff'
OFFWHITE = '#dcd4c7'
NULL = '#00000000'
WHITE = '#FFFFFF00'
ACCENT_DARK = '#990000'
ACCENT_LIGHT = '#FFCC00'

#: Standard color for font, gridlines, etc.
BASE = HILITE

#: Background color for plots
BG = DARK


plx = np.array(mpl.colormaps.get('plasma').colors)
lint = np.array(np.linspace(0, 1, num=plx.shape[0]), ndmin=2).transpose()

#: Default colormap for monopolar data
CM_MONO = LinCM.from_list('mono', np.hstack((plx, lint)))


scopeColors = ['#ffff00', '#00ffff', '#990000', '#00ff00',
               '#ff0000', '#0000ff', '#ff8000', '#8000ff',
               '#ff0080', '#0080ff']
scopeColorsLite = ['#ffcc00', '#17becf', '#990000', '#2ca02c',
                   '#1c2a99', '#d62728', '#ff7f0e', '#9467bd',
                   '#990000', '#7f7f7f']

styleScope = {
    'axes.prop_cycle': mpl.cycler('color', scopeColors)}

styleScope2 = {
    'axes.prop_cycle': mpl.cycler('color', scopeColorsLite)}


def make_style_dict(fgColor, bgColor):
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
    bgCategories = ['axes.facecolor', 'figure.edgecolor',
                    'figure.facecolor', 'savefig.edgecolor',
                    'savefig.facecolor']

    fgCategories = ['axes.edgecolor', 'axes.labelcolor',
                    'boxplot.boxprops.color', 'boxplot.capprops.color',
                    'boxplot.flierprops.color',
                    'boxplot.flierprops.markeredgecolor',
                    'boxplot.whiskerprops.color', 'grid.color', 'lines.color',
                    'patch.edgecolor', 'text.color', 'xtick.color',
                    'ytick.color', ]

    styleDict = {
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

    global FAINT

    FAINT = mpl.colors.to_rgba(BASE, MESH_ALPHA)

    return styleDict


def use_dark_style():
    """
    Switch to dark-mode visualizations (suitable for screen).

    Returns
    -------
    None.

    """
    global BASE, CM_MONO, CM_BIPOLAR, MESH_ALPHA, BG
    MESH_ALPHA = 0.25
    BASE = HILITE
    BG = DARK

    plt.style.use(make_style_dict(fgColor=OFFWHITE, bgColor=DARK))
    plt.style.use(styleScope2)
    plt.style.use({'font.size': 10})

    plx = np.array(mpl.colormaps.get('plasma').colors)
    lint = np.array(np.linspace(0, 1, num=plx.shape[0]), ndmin=2).transpose()
    CM_MONO = LinCM.from_list('mono', np.hstack((plx, lint)))

    biArray = np.vstack(([0, 0, 1, 1],
                         mpl.colors.to_rgba(DARK, alpha=0.5),
                         [1, 0, 0, 1]))

    CM_BIPOLAR = LinCM.from_list(
        'bipolar', biArray)

    pvtheme = setup_pv_theme(themes.DarkTheme())
    pvtheme.edge_color = pv.colors.Color(HILITE, opacity=0.01)
    pvtheme.font.color = pv.colors.Color(OFFWHITE)
    pvtheme.name = 'xcellDark'

    pv.global_theme.load_theme(pvtheme)


def use_light_style():
    """
    Switch to light-mode visualizations (suitable for print).

    Returns
    -------
    None.

    """
    global BASE, CM_MONO, CM_BIPOLAR, MESH_ALPHA, BG
    MESH_ALPHA = 0.5
    BASE = DARK
    BG = WHITE

    plt.style.use(make_style_dict(fgColor=DARK, bgColor=WHITE))
    plt.style.use(styleScope2)
    plt.style.use({'font.size': 11})

    cm = mpl.colormaps.get('YlOrRd')
    CM_MONO = cm.copy()

    CM_BIPOLAR = mpl.colormaps.get('seismic').copy()

    pvtheme = setup_pv_theme(themes.DarkTheme())
    pvtheme.edge_color = pv.colors.Color(DARK, opacity=MESH_ALPHA)
    pvtheme.font.color = pv.colors.Color(DARK)
    pvtheme.name = 'xcellLight'

    pv.global_theme.load_theme(pvtheme)


def setup_pv_theme(theme):
    """
    Set PyVista to match current xcell theme.

    Parameters
    ----------
    theme : PyVista theme
        Built-in theme to use as a starting point

    Returns
    -------
    theme : PyVista theme
        Customized theme
    """
    theme.show_edges = False
    theme.axes.box = True
    theme.axes.show = True
    theme.jupyter_backend = 'server'
    theme.color_cycler = 'matplotlib'
    theme.transparent_background = True
    theme.background = pv.colors.Color(BG, opacity=0)
    theme.colorbar_orientation = 'vertical'

    return theme


def recolor_svg(fname, toLight=True):
    """
    Post-process SVG to change color scheme.

    Parameters
    ----------
    fname : str
        File name.
    toLight : bool, optional
        Set conversion direction from dark to light. The default is True.

    Returns
    -------
    None.

    """
    fstr = open(fname)
    rawtxt = fstr.read()
    fstr.close()
    darkColors = [DARK, OFFWHITE, HILITE]
    liteColors = ['#FFFFFF', DARK, ACCENT_DARK]

    darkColors.extend(scopeColors)
    liteColors.extend(scopeColorsLite)

    if toLight:
        origColors = darkColors
        newColors = liteColors
        tag = '-lite'

    else:
        origColors = liteColors
        newColors = darkColors
        tag = '-dark'

    for oldC, nuC in zip(origColors, newColors):
        regex = re.compile(re.escape(oldC), re.IGNORECASE)
        rawtxt = regex.sub(nuC, rawtxt)

    name, ext = path.splitext(fname)

    newfile = open(name+tag+ext, 'w')

    newfile.write(rawtxt)
    newfile.close()


use_dark_style()
