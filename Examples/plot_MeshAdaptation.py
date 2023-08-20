#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptation parameters
================================

Illustrates how adaptation parameters affect the generated mesh

"""

import Common_nongallery
import numpy as np
import xcell as xc
import matplotlib.pyplot as plt

dmin = 2
dmax = 6

# keep animations in list so they show up in Sphix gallery
animations = []

study, setup = Common_nongallery.makeSynthStudy("adaptationDemos")
plotprefs = {"colorbar": False, "barRatio": [7, 1], "labelAxes": False}

sweepval = 1 - abs(np.linspace(-1, 1, 20))

for swept in ["density", "depth"]:
    if swept == "density":
        vrange = 0.5 * sweepval
    elif swept == "depth":
        # vrange=dmin+np.array((dmax-dmin)*sweepval,dtype=int)
        vrange = np.concatenate((np.arange(dmin, dmax + 1), np.arange(dmax - 1, dmin - 1, -1)))
    tvec = np.linspace(0, 1, vrange.shape[0])

    tdata = {"x": tvec, "y": vrange, "ylabel": swept, "unit": "", "style": "dot"}

    with plt.rc_context({"figure.figsize": [4.5, 3.75], "font.size": 10, "figure.dpi": 144}):
        img = xc.visualizers.SingleSlice(None, study, timevec=tvec, tdata=tdata, prefs=plotprefs)
        img.axes[2].set_xticks([])

        lastdepth = -1
        for val in vrange:
            if swept == "density":
                density = val
                max_depth = dmax
            elif swept == "depth":
                max_depth = val
                density = 0.2

            metric = xc.general_metric

            setup.make_adaptive_grid(
                np.zeros((1, 3)),
                max_depth=np.array(max_depth, ndmin=1),
                min_l0_function=metric,
                coefs=np.ones(1) * 2 ** (-max_depth * density),
            )
            setup.finalize_mesh()

            img.add_simulation_data(setup, append=True)

        animations.append(img.animate_study(fname=swept, fps=5.0))
# sphinx_gallery_thumbnail_number = 2
