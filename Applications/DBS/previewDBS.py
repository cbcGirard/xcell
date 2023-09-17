#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:42:08 2023

@author: benoit
"""
# %%
import xcell as xc
import pyvista as pv
import pickle
import os
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
from matplotlib import colormaps as mcmap
from matplotlib.colors import SymLogNorm

from makeStim import getSignals
import electrode
import argparse
import cmasher as cmr

cli = argparse.ArgumentParser()
cli.add_argument('-D', '--maxDepth', type=int,
                 help='Max depth', default=16)
cli.add_argument('-r', '--remote', action='store_true')
cli.add_argument('-a', '--adaptive',
                 action='store_true')
cli.add_argument('-i', '--iso', action='store_true')
cli.add_argument('-L', '--log', action='store_true')
cli.add_argument('-l', '--liteMode',
                 action='store_true')


args = cli.parse_args()
preview = not args.remote
Depth = args.maxDepth
adaptive = args.adaptive
viewIso = args.iso
symlog = args.log
liteMode = args.liteMode

# overrides
Depth = 16
adaptive = True
symlog = True
amplitude = False
# liteMode = True
preview = False

if liteMode:
    xc.colors.use_light_style()
else:
    xc.colors.use_dark_style()

fontsize = 24
pv.global_theme.font.size = fontsize
pv.global_theme.font.label_size = fontsize
pv.global_theme.font.title_size = fontsize
pv.global_theme.line_width = 15.0


sigma_0 = 0.3841
Coef = 0.1
gensInMicro = 4  # expected to converge at this value


showCurrents = False
# adaptive = True

# needed for remote execution
if args.remote:
    pv.start_xvfb()

# hack for reloading regions
rsave = pv.read('composite.vtm')
regions = xc.io.Regions()
regions['Conductors'] = rsave['Conductors']


# for later clipping to XY coords of hippocampus
ROI = regions['Conductors'].bounds[:4]

# set sim boundaries to 2x hippocampus bounding box
bbox = 2*np.max(np.abs(ROI))*np.concatenate((-np.ones(3), np.ones(3)))

dmicro = 40e-6
# dmicro = 5e-4

microRows = 6
microCols = 4
dualRows = True

nmicro = microRows*microCols

# override depth to ensure multiple elements within microelectrodes
# Depth = gensInMicro+int(np.ceil(np.log2(np.max(bbox)/dmicro)))

fstem = 'full-d%dx%d' % (Depth, int(Coef*10))

if adaptive:
    fstem = 'adaptive-'+fstem

study = xc.Study(os.path.join(os.getcwd(), fstem), bbox)

sim = study.new_simulation()

body, microElectrodes, macroElectrodes = electrode.makeDBSElectrode()


channels = getSignals(nmicro, step=50e-3)
electrode.insertInRegion(regions, body, microElectrodes, macroElectrodes)
electrode.addStandardStim(sim, microElectrodes, macroElectrodes)
# tvec = np.linspace(0, 2)
# tvec = [0]
tlist = [c.times for c in channels]
tlist.append([2.])
tvec = np.unique(np.concatenate(tlist))


# preview simulation boundary, tissue and electrode arrangement
rdisp = regions.copy()
rc = rdisp['Conductors']

rdisp['Insulators'][0] = rdisp.toPlane()['Insulators'][0]
rdisp['Conductors'] = rc.clip('z')
simbb = bbox[xc.io.TO_PV_BBOX_ORDER]

linecolor = pv.Color(xc.colors.BASE)

windowInch = np.array([7.2, 5.4])
DPI = 200
windowPx = np.array(DPI*windowInch, dtype=int)

p = xc.visualizers.PVScene(figsize=windowInch, dpi=DPI, off_screen=args.remote)
p.setup(rdisp, opacity=1.,
        # rdisp['Conductors'],
        simData='sigma',
        scalar_bar_args={'title': r'$\sigma[S/m]$',
                         'title_font_size': 9,
                         'label_font_size': 9,
                         'fmt': '%.2f'}
        )
p.add_ruler(pointa=(ROI[0], 1.2*ROI[2], 0.),
            pointb=(ROI[0]+5e-3, 1.2*ROI[2], 0.),
            font_size_factor=1.,
            tick_color='black',
            label_color='black',
            title='5 mm',
            # label_format='%''.0e',
            # title='Length [m]',
            tick_length=15,
            number_labels=0,
            show_labels=False,
            number_minor_ticks=0
            )
p.planeview(1.1*np.array(ROI))


# highligt small electrodes
microcenters = np.array([m.center for m in regions['Electrodes'][:nmicro]])
p.add_mesh(pv.wrap(microcenters), color='gold',
           render_points_as_spheres=True, point_size=10.)
study.save_pv_image(p, 'setup2.png', window_size=windowPx)

if preview:
    p.show()
else:
    p.close()

print(fstem)

# %% Show currents

if preview and showCurrents:
    p = xc.visualizers.PVScene(time=tvec)
    p.camera.focal_point = np.zeros(3)
    p.camera.position = 2 * \
        np.linalg.norm(regions['Conductors'].bounds[::2]
                       )*np.array([-1., -1., 1.])

    p.open_movie(fstem+'Currents.mp4')
    ROI = regions['Conductors'].bounds[:4]

    for m in regions['Electrodes']:
        m.point_data['voltage'] = 0

    VBAR = {
        'vertical': True,
        'height': 0.8,
        'position_x': 0.0,
        'position_y': 0.2
    }

    pts = pv.wrap(
        np.array([src.geometry.center for src in sim.current_sources]))
    pts.point_data['voltage'] = 0

    p.setup(regions)
    p.add_mesh(pts,
               clim=150e-6*np.array([-1, 1]),
               show_edges=True, cmap=xc.colors.CM_BIPOLAR,
               # log_scale=True,
               scalar_bar_args=VBAR, render_points_as_spheres=True,
               point_size=10)
    # p.planeview(ROI)

    for ii in tqdm.trange(len(tvec)):
        t = tvec[ii]
        if ii+1 == len(tvec):
            tnext = 2.
        else:
            tnext = tvec[ii+1]

        sim.current_time = t
        currentV = [src.value.getCurrentValue(t) for src in sim.current_sources]
        pts['voltage'] = currentV

        # for msh, sig in zip(regions['Electrodes'], channels):
        #     msh['voltage'] = sig.getCurrentValue(t)*np.ones_like(msh['voltage'])

        p.setTime(t)
        p.write_frame()

        # while t < tnext:
        #     p.setTime(t)
        #     p.write_frame()
        #     t += 0.001

    p.close()

for src in sim.current_sources:
    src.value.reset()


# %% Simulate
refs = np.array([src.geometry.center for src in sim.current_sources[:nmicro]])

# refs = np.concatenate([r.points for r in regions['Conductors']])
# coefs = np.ones(refs.shape[0])

if adaptive:
    depthmin = 6
    depthspan = Depth - depthmin  # adaptive range

    depthvec = np.repeat(depthmin, refs.shape[0])

else:
    depthvec = np.repeat(Depth, refs.shape[0])

coefs = 2**(-Coef*depthvec)

sim.make_adaptive_grid(ref_pts=refs, max_depth=depthvec,
                     min_l0_function=xc.general_metric,
                     coefs=coefs, coarsen=False)
sim.finalize_mesh(sigma_mesh=regions, default_sigma=sigma_0)
sim.set_boundary_nodes()

# sim.start_timing('Assign sigma')
# regions.assign_sigma(sim.mesh, default_sigma=sigma_0)
# sim.log_time()
vmesh = xc.io.to_vtk(sim.mesh)
vmesh.point_data['voltage'] = 0

vslice = vmesh.slice(normal='z')

# save meshes for later
vmesh.save(os.path.join(fstem, 'sim0.vtk'))
vslice.save(os.path.join(fstem, 'Slice.vtk'))

p = xc.visualizers.PVScene(time=tvec)
p.setup(regions.toPlane(), mesh=vslice,
        simData='sigma', show_edges=True)
p.planeview(ROI)
study.save_pv_image(p, 'meshed', title='Meshed geometry')

if preview:
    p.show()

volts = []
voltSlices = []


# %%
timer = tqdm.trange(len(tvec), desc='Simulating', postfix='')
changed = False
infos = []
for ii in timer:
    t = tvec[ii]
    # t = tvec[4]
    sim.current_time = t

    if adaptive and ii != 0:
        depths = []
        for src in sim.current_sources[:nmicro]:
            if src.get_value_at_time(t) == 0:
                depths.append(depthmin)
            else:
                depths.append(Depth)
        depthvec = np.array(depths)

        coefs = 2**(-Coef*depthvec)

        changed = sim.make_adaptive_grid(ref_pts=refs, max_depth=depthvec,
                                       min_l0_function=xc.general_metric,
                                       coefs=coefs)
        if changed:

            sim.finalize_mesh(sigma_mesh=regions, default_sigma=sigma_0)
            sim.set_boundary_nodes()

        # else:
        #     # sim.node_role_table[sim.node_role_table == 2] == 0
        #     sim.finalize_mesh()
        #     sim.set_boundary_nodes()

    v = sim.solve()

    # MUST use v.copy; otherwise, results in list of the last timestep's result at every time!
    volts.append(v.copy())

    # vmesh['voltage'] = v.copy()
    # tempSlice = vmesh.slice(normal='z')
    # voltSlices.append(tempSlice['voltage'])

    # if adaptive and changed:
    sim.iteration += 1
    sim.name = 'sim%d' % sim.iteration
    vmesh = xc.io.to_vtk(sim.mesh)
    vmesh.save(os.path.join(fstem, '%s.vtk' %
               sim.name))

    study.log_current_simulation()

    stepinfo = {'meshName': sim.name}
    twall = 0
    tcpu = 0
    for l in sim.step_logs:
        stepinfo[l.name+' [Wall]'] = l.durationWall
        stepinfo[l.name+' [CPU]'] = l.durationCPU
        stepinfo[l.name+' [memory]'] = l.memory

        twall += l.durationWall
        tcpu += l.durationCPU

    stepinfo['Total time [Wall]'] = twall
    stepinfo['Total time [CPU]'] = tcpu

    infos.append(stepinfo)

    sim.step_logs = []
    timer.set_postfix_str('Vrange=[%.2g, %.2g]' % (min(v), max(v)))

    # print(sim.RHS[sim.RHS != 0])

    if np.max(np.abs(v)) > 100:
        exit
    # vmesh.point_data['v'] = v
    # p = xc.visualizers.PVScene()
    # p.setup(regions.toPlane(), vmesh.slice('z'), simData='v',
    #         show_edges=True)
    # p.planeview(ROI)
    # p.show()

study.save(volts, fstem+'Volts')
study.save(voltSlices, fstem+'VoltSlices')
study.save(infos, fstem+'info')

# %% Separate plotting logic

moviename = 'voltage'
meshname = 'sim0'

plotTime = tvec
# plotTime = None

data_category = 'voltage'
# data_category = 'sigma'
# moviename = 'sigma'
# symlog = False

# reloading pickled data
if not 'infos' in dir():
    infos = study.load(fstem+'info')
if not 'volts' in dir():
    volts = study.load(fstem+'Volts')

if viewIso:
    mesh = vmesh
    data = volts
    moviename += '-iso'
else:
    data = []
    recon = tqdm.trange(len(infos), desc='Reconstructing')
    for ii in recon:
        meshname = infos[ii]['meshName']
        v = volts[ii]

        msh = pv.read((os.path.join(fstem, meshname+'.vtk')))
        if ii == 0:
            if data_category == 'sigma':
                msh.cell_data[data_category] = np.zeros(msh.n_cells)
            else:
                msh.point_data[data_category] = np.zeros(msh.n_points)
            vmesh = msh.copy()
            mesh = msh.copy()
        else:
            if data_category == 'sigma':
                regions.assign_sigma(msh, default_sigma=sigma_0)
            else:
                msh.point_data[data_category] = v.copy()
        # print('%d\t%d' % (len(v), msh.n_points))

        slic = msh.slice(normal='z')

        data.append(slic[data_category])

    # vmesh.point_data['voltage'] = 0
    mesh = vmesh.slice(normal='z')
    # data = mesh['voltage']

dvec = []
for d in data:
    dvec.extend(d)

vrange = [np.min(dvec), np.max(dvec)]
vspan = np.sign(vrange)*np.max(np.abs(vrange))


pargs = {}
VBAR = {
    'vertical': True,
    'height': 0.75,
    'position_x': 0.05,
    'position_y': 0.2,
    'title_font_size': 20,
    'label_font_size': 20,
}

if data_category == 'sigma':
    cma = xc.colors.CM_MONO
    clim = np.array([0., 0.5])

else:
    if symlog:
        moviename += '-log'
        symmer = SymLogNorm(vspan[1]/100, vmax=vspan[1], vmin=vspan[0])
        clim = (0., 1.)
        cma = xc.colors.CM_BIPOLAR
        # opacity = np.ones(11)
        # opacity[11//2] = 0.
        # pargs['opacity'] = opacity
        cbarfig = xc.visualizers.makeSoloColorbar(None,
                                                  cmap=cma, norm=symmer, unit='V')
    elif amplitude:
        clim = max(np.abs(vrange))*np.array([1e-3, 1.])
        cma = 'magma_r'
        pargs['log_scale'] = True
        VBAR['title'] = 'Amplitude [v]'
    else:
        clim = vrange
        cma = 'seismic'


p = xc.visualizers.PVScene(
    time=plotTime, figsize=windowInch, dpi=DPI, off_screen=True)

# p.open_movie(fstem+'.mp4')
study.make_pv_movie(p, filename=moviename)


# cma = mcmap['bwr']
# cma = xc.colors.CM_BIPOLAR
# cma=mcmap['seismic']
cma = xc.colors.scoop_cmap(cmr.guppy_r)


if viewIso:
    p.setup(regions)
else:
    p.setup(regions.toPlane(), mesh=mesh, simData=data_category, clim=clim,
            show_edges=True, cmap=cma, scalar_bar_args=VBAR,
            show_scalar_bar=(not symlog),
            lighting=False,
            **pargs)

# p.add_mesh(mesh, scalars='voltage', clim=vrange,
#            show_edges=True, cmap=cma,
#            scalar_bar_args=VBAR, symlog=symlog)

# orient for 2d image
if not viewIso:
    p.planeview(ROI)


# mesh['voltage'] = data[121]

# if not preview:
# p.show(auto_close=False)

viz = tqdm.trange(len(tvec)-1, desc='Animating')
# viz = [0, 14, 43, 59, 79, 102, ]
for ii in viz:
    t = tvec[ii]
    if ii+1 == len(tvec):
        tnext = 2.
    else:
        tnext = tvec[ii+1]

    info = infos[ii]
    if info['meshName'] != meshname:
        fullmesh = pv.read(
            os.path.join(fstem,
                         info['meshName']+'.vtk'))
        meshname = info['meshName']
        if viewIso:
            newmesh = fullmesh
        else:
            newmesh = fullmesh.slice(normal='z')

        mesh.copy_from(newmesh)
        p.edgeMesh.copy_from(newmesh)

    if symlog:
        mesh[data_category] = symmer(data[ii])
    elif amplitude:
        mesh[data_category] = np.abs(data[ii])
    else:
        mesh[data_category] = data[ii]

    # p.setTime(t)
    # p.write_frame()
    # p.screenshot('f%03d.png' % ii)

    while t < tnext:
        p.setTime(t)
        p.write_frame()
        # if liteMode:
        study.save_pv_image(p,
                          os.path.join(moviename,
                                       moviename+'frame%03d.png' % ii))
        t += 0.005
        break

if liteMode:
    study.save_plot(cbarfig,
                   os.path.join(moviename, moviename+'colorbar'))

p.close()
