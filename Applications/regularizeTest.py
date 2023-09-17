#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:02:59 2022
Regularization tests
@author: benoit
"""


import numpy as np
import numba as nb
import xcell as xc
import matplotlib.pyplot as plt


meshtype = "adaptive"
# study_path='Results/studyTst/miniCur/'#+meshtype
study_path = "Results/studyTst/regularization/"

xmax = 1e-4
sigma = np.ones(3)


vMode = False
showGraphs = False
generate = False
saveGraphs = False

vsrc = 1.0
isrc = vsrc * 4 * np.pi * sigma * 1e-6

bbox = np.append(-xmax * np.ones(3), xmax * np.ones(3))


study = xc.Study(study_path, bbox)

min_l0 = 1e-6
rElec = 1e-6

lastNumEl = 0
meshTypes = ["adaptive", "uniform"]


if generate:
    for max_depth in range(2, 20):
        for regularize in range(2):
            # for max_depth in range(2,10):
            # if meshtype=='uniform':
            #     max_depth=var
            # else:
            l0Param = 2 ** (-max_depth * 0.2)
            # l0Param=0.2

            setup = study.new_simulation()
            setup.mesh.element_type = "Admittance"
            setup.meshtype = meshtype
            setup.mesh.minl0 = 2 * xmax / (2**max_depth)
            setup.ptPerAxis = 1 + 2**max_depth

            geo = xc.geometry.Sphere(np.zeros, rElec)

            if vMode:
                srcMag = 1.0
                srcType = "Voltage"
                setup.add_voltage_source(xc.signals.Signal(1.0), geo)
            else:
                srcMag = 4 * np.pi * sigma[0] * rElec
                setup.add_current_source(xc.signals.Signal(1.0), geo)
                srcType = "Current"

            if meshtype == "uniform":
                newNx = int(np.ceil(lastNumEl ** (1 / 3)))
                nX = newNx + newNx % 2
                setup.make_uniform_grid(newNx + newNx % 2)
                print("uniform, %d per axis" % nX)
            else:

                def metric(coord, l0Param=l0Param):
                    r = np.linalg.norm(coord)
                    val = l0Param * r  # 1/r dependence
                    # val=(l0Param*r**2)**(1/3) #current continuity
                    # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
                    # if val<rElec:
                    #     val=rElec

                    if (r + val) < rElec:
                        val = rElec / 2
                    return val

                # def metric(coord):
                #     r=np.linalg.norm(coord)
                #     # val=l0Param*r #1/r dependence
                #     val=(1e-7*r**2)**(1/3) #current continuity
                #     # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
                #     # if val<rElec:
                #     #     val=rElec

                #     if (r+val)<rElec:
                #         val=rElec/2
                #     return val

                setup.make_adaptive_grid(metric, max_depth)

            def boundary_function(coord):
                r = np.linalg.norm(coord)
                return rElec / (r * np.pi * 4)

            setup.finalize_mesh(regularize)

            setup.set_boundary_nodes(boundary_function)

            # v=setup.solve()
            v = setup.solve(None, 1e-9)

            setup.getMemUsage(True)
            setup.print_total_time()

            setup.start_timing("Estimate error")
            errEst, _, _, _ = setup.calculate_errors()  # srcMag,srcType,showPlots=showGraphs)
            print("error: %g" % errEst)
            setup.log_time()

            study.log_current_simulation(
                ["Error", "k", "Depth", "Rregularized?"], [errEst, l0Param, max_depth, str(bool(regularize))]
            )
            study.save_simulation(setup)
            lastNumEl = len(setup.mesh.elements)

            # ax=xc.new3dPlot( bbox)
            # xc.show_3d_edges(ax, coords, setup.mesh.edges)
            # break

            # fig=plt.figure()
            # xc.centerSlice(fig, setup)

            # if saveGraphs:
            #     study.makeStandardPlots()


# aniGraph=study.animatePlot(xc.error2d,'err2d')
# aniGraph=study.animatePlot(xc.ErrorGraph,'err2d_adaptive',["Mesh type"],['adaptive'])
# aniGraph2=study.animatePlot(xc.error2d,'err2d_uniform',['Mesh type'],['uniform'])
# aniImg=study.animatePlot(xc.centerSlice,'img_mesh')
# aniImg=study.animatePlot(xc.SliceSet,'img_adaptive',["Mesh type"],['adaptive'])
# aniImg2=study.animatePlot(xc.centerSlice,'img_uniform',['Mesh type'],['uniform'])


fig = plt.figure()
plotters = [xc.ErrorGraph, xc.SliceSet, xc.CurrentPlot, xc.CurrentPlot]
ptype = ["ErrorGraph", "SliceSet", "CurrentShort", "CurrentLong"]
regNames = ["Initial", "Regularized"]

for mt in range(2):
    for ii, p in enumerate(plotters):
        plt.clf()
        if ii == 3:
            plotr = p(fig, study, fullarrow=True)
        else:
            plotr = p(fig, study)

        isreg = bool(mt)
        plotr.get_study_data(filter_categories=["Regularized?"], filter_values=[isreg])

        name = ptype[ii] + "_" + regNames[mt]

        ani = plotr.animate_study(name)


# plotE=xc.ErrorGraph(plt.figure(),study)#,{'showRelativeError':True})
# plotE.get_study_data()
# aniE=plotE.animate_study()


# plotS=xc.SliceSet(plt.figure(),study)
# plotS.get_study_data()
# aniS=plotS.animate_study()


# plotC=xc.CurrentPlot(plt.figure(),study)
# plotC.get_study_data()
# aniC=plotC.animate_study()

xc.grouped_scatter(
    study.study_path + "log.csv",
    x_category="Number of elements",
    y_category="Error",
    group_category="Regularized?",
)
nufig = plt.gcf()
study.save_plot(nufig, "AccuracyCost", ".eps")
study.save_plot(nufig, "AccuracyCost", ".png")
