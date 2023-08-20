#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:56:54 2021

@author: benoit
"""

import numpy as np
import numba as nb
import xcell as xc
import matplotlib.pyplot as plt
import pickle

meshtype = "adaptive"
# study_path='Results/studyTst/miniCur/'#+meshtype
datadir = "/home/benoit/smb4k/ResearchData/Results/"  # +meshtype
# study_path=datadir+'post-renumber/'

sigma = np.ones(3)

X = 1000
V = 1
R = 2
study_path = datadir + "errorMetrics/"

generate = True


def run(X, V, R, generate):
    xmax = X * 1e-6

    vMode = False

    bbox = np.append(-xmax * np.ones(3), xmax * np.ones(3))

    study = xc.Study(study_path, bbox)

    min_l0 = 1e-6
    rElec = R * 1e-6

    lastNumEl = 0
    lastNx = 0

    tstVals = [None]
    tstCat = "Power"

    errdicts = []

    if generate:
        # for var in np.linspace(0.1,0.7,15):
        for meshnum, max_depth in enumerate(range(3, 16)):
            for tstVal in tstVals:
                # meshtype=tstVal
                # element_type=tstVal

                element_type = "Admittance"
                # for vMode in tstVals:
                # for max_depth in range(2,10):
                # if meshtype=='uniform':
                #     max_depth=var
                # else:
                l0Param = 2 ** (-max_depth * 0.2)
                # l0Param=0.2

                setup = study.new_simulation()
                setup.mesh.element_type = element_type
                setup.meshtype = meshtype
                setup.meshnum = meshnum

                geo = xc.geometry.Sphere(center=np.zeros(3), radius=rElec)

                if vMode:
                    srcMag = 1.0
                    srcType = "Voltage"
                    source = xc.signals.Signal(srcMag)
                    setup.add_voltage_source(source, geometry=geo)

                else:
                    srcMag = 4 * np.pi * sigma[0] * rElec * 10
                    source = xc.signals.Signal(srcMag)
                    setup.add_current_source(srcMag, geometry=geo)
                    srcType = "Current"

                # if meshtype=='equal elements':
                if meshtype == "uniform":
                    newNx = int(np.ceil(lastNumEl ** (1 / 3)))
                    nX = newNx + newNx % 2
                    setup.make_uniform_grid(newNx + newNx % 2)
                    print("uniform, %d per axis" % nX)
                elif meshtype == r"equal $l_0$":
                    setup.make_uniform_grid(lastNx)
                else:
                    metric = xc.makeExplicitLinearMetric(max_depth, meshdensity=0.2)

                    setup.make_adaptive_grid(metric, max_depth)

                setup.mesh.element_type = element_type
                setup.finalize_mesh()
                # if asDual:
                #     setup.finalizeDualMesh()
                # else:
                #     setup.finalize_mesh(regularize=False)

                coords = setup.mesh.node_coords
                #

                def boundary_function(coord):
                    r = np.linalg.norm(coord)
                    return rElec / (r * np.pi * 4)

                # setup.insert_sources_in_mesh()
                setup.set_boundary_nodes(boundary_function)
                # setup.get_edge_currents()

                # v=setup.solve()
                v = setup.solve(None, 1e-9)

                setup.apply_transforms()

                setup.getMemUsage(True)
                errEst, errVec, vAna, sorter, r = setup.calculate_errors()
                print("error: %g" % errEst)

                sse = np.sum(errVec**2)
                sstot = np.sum((v - np.mean(v)) ** 2)
                ss = np.sum((vAna - np.mean(vAna)) ** 2)
                FVU = sse / sstot

                _, basicAna, basicErr, _ = setup.estimate_volume_error(basic=True)
                _, advAna, advErr, _ = setup.estimate_volume_error(basic=False)

                errBasic = sum(basicErr) / sum(basicAna)
                errAdv = sum(advErr) / sum(advAna)

                numel = len(setup.mesh.elements)

                # power = setup.getPower()

                power = xc.misc.estimate_power(setup.node_voltages, setup.edges, setup.conductances)
                print("power: " + str(power))
                study.log_current_simulation(
                    ["AreaError", "basicVol", "advVol", "SSE", "SStot", "SS", "FVU", "power"],
                    [errEst, errBasic, errAdv, sse, sstot, ss, FVU, power],
                )
                study.save_simulation(setup)

                errdict = xc.misc.get_error_estimates(setup)
                # errdict['densities']=density
                errdict["depths"] = max_depth
                errdict["numels"] = numel
                errdicts.append(errdict)

        dlist = xc.misc.transpose_dicts(errdicts)
        pickle.dump(dlist, open(study_path + "errMets.p", "wb"))

    else:
        dlist = pickle.load(open(study_path + "errMets.p", "rb"))

    for met in ["FVU", "FVU2", "powerError", "int1", "int3"]:
        plt.loglog(dlist["numels"], dlist[met], label=met)

    plt.legend()
    plt.xlabel("Number of elements")
    plt.title("%dum domain, %dV-equivalent %dum source" % (X, V, R))

    study.save_plot(plt.gcf(), "%du-%dv-%dr" % (X, V, R), ".png")
    study.save_plot(plt.gcf(), "%du-%dv-%dr" % (X, V, R), ".eps")
    plt.close("all")


for R in [1, 10]:
    for X in [100, 1000, 10000]:
        for V in [1, 10]:
            run(X, V, R, generate)
