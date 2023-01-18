#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience functions."""

import numpy as np


def iSourcePower(isrc, radius, sigma):
    if type(sigma) == np.ndarray:
        sigma = np.linalg.norm(sigma)

    rInf = 1/(4*np.pi*sigma*radius)
    power = rInf*isrc**2
    return power


def vSourceIntegral(vsrc, radius, rmax):
    return radius*vsrc*(1+np.log(rmax/radius))


def iSourceIntegral(isrc, radius, rmax, sigma):
    vsrc = isrc/(4*np.pi*sigma)
    return vSourceIntegral(vsrc, radius, rmax)


def estimatePower(voltages, edges, conductances):
    dv = voltages[edges]
    v2 = np.diff(dv, axis=1)**2
    return np.dot(v2.squeeze(), conductances)


def getErrorEstimates(simulation):
    errSummary, err, vAna, sorter, r = simulation.calculateErrors()
    intErr = np.trapz(err[sorter], r[sorter])

    v = vAna-err

    sse = np.sum(err**2)
    # sstot=np.sum((v-np.mean(v))**2)
    sstot = np.sum((vAna-np.mean(vAna))**2)
    # FVU=sse/sstot
    FVU = sse/sstot

    elV, elAna, elErr, _ = simulation.estimateVolumeError(basic=True)

    volErr = sum(np.abs(elErr))
    volAna = sum(np.abs(elAna))
    vol = volErr/volAna

    # powerSim = estimatePower(simulation.nodeVoltages,
    #                          simulation.edges,
    #                          simulation.conductances)

    # powerTrue = iSourcePower(simulation.currentSources[0].value,
    #                          simulation.currentSources[0].radius,
    #                          simulation.mesh.tree.sigma)

    # powerErr = abs(powerTrue-powerSim)/powerTrue

    absErr = abs(err)
    data = {
        'max': max(absErr),
        'min': min(absErr),
        'avg': np.mean(absErr),
        'int1': errSummary,
        'int3': vol,
        'volErr': volErr,
        'volAna': volAna,
        'intErr': intErr,
        'intAna': intErr/errSummary,
        'SSE': sse,
        'SSTot': sstot,
        'FVU': FVU,
        # 'powerSim': powerSim,
        # 'powerTrue': powerTrue,
        # 'powerError': powerErr,
    }

    return data


def getSquares(err, vAna):
    SSE = np.sum((err)**2)
    SSTot = np.sum((vAna-np.mean(vAna))**2)

    return SSE, SSTot


def transposeDicts(dictList):
    arrayDict = {}

    for d in dictList:
        emptyDict = len(arrayDict.keys()) == 0
        for k, v in zip(d.keys(), d.values()):
            if emptyDict:
                arrayDict[k] = [v]
            else:
                arrayDict[k].append(v)

    return arrayDict


def FVU(analyticVals, err):
    # err=simVals-analyticVals

    SSE = np.sum(err**2)
    SSTot = np.sum((analyticVals-np.mean(analyticVals))**2)

    FVU = SSE/SSTot

    return FVU
