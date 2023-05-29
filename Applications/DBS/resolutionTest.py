#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:17:33 2023

@author: benoit
"""

import electrode
import xcell
import pyvista as pv
import numpy as np
import os
from tqdm import trange
import pickle
import matplotlib.pyplot as plt
from makeStim import getSignals


preview = True
activeElec = 0
Coef = 0.1
sigma_0 = 0.3841

tol = 0.05
singles = False

minD = 6
# geoms=['sphere', 'cylinder', 'disk']
geoms = ['disk']
# elecGeom = 'disk'

if not singles:
    signals = getSignals(24)
    tvals = np.unique(
        [t for s in signals for t, v in zip(s.times, s.values) if v > 0])


xcell.colors.useLightStyle()

# needed for remote execution
if not preview:
    pv.start_xvfb()

# hack for reloading regions
rsave = pv.read('../../Examples/Geometry/composite.vtm')

# %%
if singles:
    elecSelector = trange(24, desc='Electrode number')
else:
    elecSelector = trange(len(tvals), desc='Pulse number')
for activeElec in elecSelector:
    for elecGeom in geoms:
        regions = xcell.io.Regions()
        regions['Conductors'] = rsave['Conductors']

        # for later clipping to XY coords of hippocampus
        ROI = regions['Conductors'].bounds[:4]

        # set sim boundaries to 2x hippocampus bounding box
        bbox = 2*np.max(np.abs(ROI))*np.concatenate((-np.ones(3), np.ones(3)))

        fstem = os.path.join('sweep', '%s-%d' % (elecGeom, activeElec))

        study = xcell.SimStudy(os.path.join(os.getcwd(), fstem), bbox)

        sim = study.newSimulation()

        body, microElectrodes, macroElectrodes = electrode.makeDBSElectrode(
            microStyle=elecGeom)

        electrode.insertInRegion(
            regions, body, microElectrodes, macroElectrodes)

        if singles:
            electrode.addUnityStim(sim, microElectrodes,
                                   macroElectrodes, activeElec, amplitude=150e-6)
            theSrc = sim.currentSources[activeElec]
            refpt = np.array(theSrc.geometry.center, ndmin=2)

        else:
            electrode.addStandardStim(sim, microElectrodes, macroElectrodes)
            t = tvals[activeElec]
            sim.currentTime = t

            activeSrces = [
                src for src in sim.currentSources if src.value.getValueAtTime(t) > 0]

            refpt = np.array(
                [src.geometry.center for src in activeSrces], ndmin=2)
        # tvec = np.linspace(0, 2)
        # tvec = [0]

        # preview simulation boundary, tissue and electrode arrangement
        # if preview:
        #     p = xcell.visualizers.PVScene()
        #     p.setup(regions, opacity=0.5)
        #     p.planeview(ROI)
        #     p.add_mesh(
        #         pv.Cube(bounds=bbox[xcell.io.PV_BOUND_ORDER]), style='wireframe')
        #     p.add_mesh(pv.wrap(theSrc.geometry.center))
        #     p.show()

        vsrc = 0
        vElectrodes = []
        nSrcPts = []
        dels = []

        dprog = trange(minD, 20, postfix='', leave=False)
        for d in dprog:
            depthvec = np.array([d]*refpt.shape[0])
            coefs = 2**(-Coef*depthvec)

            sim.makeAdaptiveGrid(refPts=refpt, maxdepth=depthvec,
                                 minl0Function=xcell.generalMetric,
                                 coefs=coefs,
                                 coarsen=False)
            sim.finalizeMesh()

            sim.setBoundaryNodes()

            sim.startTiming('Assign sigma')
            regions.assignSigma(sim.mesh, defaultSigma=sigma_0)
            sim.logTime()
            v = sim.iterativeSolve()

            velec = v[np.argmax(np.abs(v))]

            if singles:
                nsrc = xcell.util.fastcount(
                    theSrc.geometry.isInside(sim.mesh.nodeCoords))
            else:
                nsrc = np.mean([xcell.util.fastcount(src.geometry .isInside(
                    sim.mesh.nodeCoords)) for src in activeSrces])

            delta = (velec-vsrc)/velec
            vElectrodes.append(velec)
            dels.append(delta)
            nSrcPts.append(nsrc)
            vsrc = velec
            dprog.set_postfix_str(
                '\nerror: %.1f%%\t%d pt in src\n' % (100*delta, nsrc), refresh=False)
            # print(sim.RHS[sim.RHS != 0])
            if len(sim.mesh.elements) > 2e6 or abs(delta) < tol:
                break

        study.save({'volts': vElectrodes,
                    'dels': dels,
                    'npt': nSrcPts
                    },
                   'vecs.xdata')


# f, axes = plt.subplots(nrows=2)

# axes[0].plot(vElectrodes)
# axes[1].semilogx(nSrcPts, dels)


# %% plot by type

for geo in geoms:
    f, axes = plt.subplots(nrows=2)
    f.suptitle(geo)

    for n in range(24):
        dat = pickle.load(
            open(os.path.join('sweep', '%s-%d/vecs.xdata.p' % (geo, n)), 'rb'))
        y = dat['volts']
        axes[0].plot(minD+np.arange(len(y)), y, label=str(n))
        axes[0].text(len(y)-1, y[-1], str(n), c='C'+str(n))
        axes[1].loglog(dat['npt'], np.abs(dat[
            'dels']), marker='*', linestyle='')

    # axes[0].legend()

# %% Show converged values in 3d

# geom='disk'

# vfinals=[pickle.load(open('sweep-%s-%d/vecs.xdata.p' %
#                             (geom, n), 'rb'))['volts'][-1] for n in range(24)]

# p=xcell.visualizers.PVScene()
# ctrs=[]

# for r, v in zip(regions[-1], vfinals):
#     ctrs.append(r.center)
#     r.point_data['v']=v
#     r.set_active_scalars('v')

# pts=pv.wrap(np.array(ctrs))
# pts.point_data['v']=vfinals

# p.setup(regions.toPlane(),
#         # mesh=regions[-1][:24],
#         mesh=pts,
#         simData='v', cmap='plasma')
# p.planeview(ROI, scale=1)
# p.show()
