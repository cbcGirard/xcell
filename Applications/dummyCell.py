#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:58:31 2022

@author: benoit
"""

from neuron import h  # , gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import ArtistAnimation
# from xcell import visualizers
import xcell
import Common
import time
import pickle


from xcell import nrnutil as nUtil
from matplotlib.lines import Line2D

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)


ring = Common.Ring(stim_delay=0)
# shape_window=h.PlotShape(True)
# shape_window.show(0)


for cell in ring.cells:
    cell.dend.nseg = 15
h.define_shape()

#


ivecs, isSphere, coords, rads = nUtil.getNeuronGeometry()

t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(20)


I = -1e-9*np.array(ivecs).transpose()
imax = np.max(np.abs(I))
cmap, norm = xcell.visualizers.getCmap(I.ravel(), forceBipolar=True)

# fig=plt.figure()
# ax=plt.gca()

fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [
                         9, 1], 'width_ratios': [9, 1]})
ax = axes[0, 0]
tax = axes[1, 0]
cax = axes[0, 1]
tax.set_xlabel('Time [ms]')
tax.yaxis.set_visible(False)
[tax.spines[k].set_visible(False) for k in tax.spines]


cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    label='Current [nA]', cax=cax)


coord = np.array(coords)
xmax = 2*np.max(np.concatenate(
    (np.max(coord, axis=0), np.min(coord, axis=0))
))

x, y, z = np.hsplit(coord, 3)

tht = np.linspace(0, 2*np.pi)

# to numpy arrays
selSphere = np.array(isSphere)
rvec = np.array(rads)


class LineDataUnits(Line2D):
    """
        Yoinked from https://stackoverflow.com/a/42972469
    """

    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def showCellGeo(axis):

    tht = np.linspace(0, 2*np.pi)
    shade = xcell.visualizers.FAINT
    polys = []
    for sec in h.allsec():
        x, y, z, r = nUtil.returnSegmentCoordinates(sec)
        coords = np.vstack((x, y, z)).transpose()

        if sec.hname().split('.')[-1] == 'soma':
            sx = x+r*np.cos(tht)
            sy = y+r*np.sin(tht)

            axis.fill(sx, sy, color=shade)
        else:
            # line=LineDataUnits(x,y, linewidth=r,color=shade)
            # axis.add_line(line)
            px = []
            py = []

            for ii in range(len(x)-1):
                p0 = coords[ii, :2]
                p1 = coords[ii+1, :2]
                d = p1-p0
                dn = r[ii]*d/np.linalg.norm(d)
                n = np.array([-dn[1], dn[0]])
                pts = np.vstack((p0+n, p1+n, p1-n, p0-n))
                # a,b=np.hsplit(pts,2)

                # px.extend(a)
                # py.extend(b)

                polys.append(pts)

                # mpl.collections.LineCollection(segments, color=shade)

    polycol = mpl.collections.PolyCollection(polys, color=shade)
    axis.add_collection(polycol)


showCellGeo(ax)
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-xmax, xmax)


arts = []
errdicts = []

lastNumEl = 0
lastI = 0


tv = np.array(t)*1e-3
tax.set_xlim(left=min(tv), right=max(tv))


study, setup = Common.makeSynthStudy('ring', xmax=xmax)
setup.currentSources = []
studyPath = study.studyPath

strat = 'depth'
dmax = 8
dmin = 4

vids = True
nskip = 10

# tdata={
#        'x': tv}

if vids:
    img = xcell.visualizers.SingleSlice(None, study,
                                        tv)

    err = xcell.visualizers.SingleSlice(None, study,
                                        tv)


for r, c, v in zip(rads, coords, I[0]):
    setup.addCurrentSource(v, c, r)

tmax = tv.shape[0]

for ii in range(0, tmax, nskip):

    t0 = time.monotonic()
    ivals = I[ii]
    tval = tv[ii]

    setup.currentTime = tval

    changed = False
    for jj in range(len(setup.currentSources)):
        ival = ivals[jj]
        setup.currentSources[jj].value = ival

        if strat == 'k':
            # k-param strategy
            density = 0.4*(abs(ival)/imax)
            print('density:%.2g' % density)

        elif strat == 'depth' or strat == 'd2':
            # Depth strategy
            scale = (dmax-dmin)*abs(ival)/imax
            dint, dfrac = divmod(scale, 1)
            maxdepth = dmin+int(dint)
            print('depth:%d' % maxdepth)

            # density=0.25
            if strat == 'd2':
                density = 0.5
            else:
                density = 0.2  # +0.2*dfrac

        elif strat == 'fixed':
            # Static mesh
            maxdepth = 5
            density = 0.2

        metrics = [xcell.makeExplicitLinearMetric(maxdepth, density)]

        changed |= setup.makeAdaptiveGrid(metrics, maxdepth)

    if changed:
        setup.meshnum += 1
        setup.finalizeMesh()

        numEl = len(setup.mesh.elements)

        setup.setBoundaryNodes()

        v = setup.iterativeSolve()
        lastNumEl = numEl
        setup.iteration += 1

        study.saveData(setup)  # ,baseName=str(setup.iteration))
    else:
        vdof = setup.getDoFs()
        v = setup.iterativeSolve(vGuess=vdof)
        # scale=ival/lastI
        # # print(scale)
        # v=setup.nodeVoltages*scale
        # setup.nodeVoltages=v

        # v=setup.solve()

    dt = time.monotonic()-t0

    lastI = ival

    study.newLogEntry(['Timestep', 'Meshnum'], [
                      setup.currentTime, setup.meshnum])

    setup.stepLogs = []

    print('%d percent done' % (int(100*ii/tmax)))

    errdict = xcell.misc.getErrorEstimates(setup)
    errdict['densities'] = density
    errdict['depths'] = maxdepth
    errdict['numels'] = lastNumEl
    errdict['dt'] = dt
    errdict['vMax'] = max(np.abs(v))
    errdicts.append(errdict)

    if vids:
        err.addSimulationData(setup, append=True)
        img.addSimulationData(setup, append=True)

lists = xcell.misc.transposeDicts(errdicts)
pickle.dump(lists, open(studyPath+strat+'.p', 'wb'))
# scat=ax.scatter(x,y,c=I[ii],cmap=cmap,norm=norm)
# title=visualizers.animatedTitle(fig, 't=%g ms'%tv[ii])
# # tbar=tax.barh(0,tv[ii])
# tbar=tax.vlines(tv[ii],0,1)
# arts.append([scat,title,tbar])


if vids:
    ani = img.animateStudy('volt-'+strat, fps=30.)
    erAni = err.animateStudy('error-'+strat, fps=30.)
