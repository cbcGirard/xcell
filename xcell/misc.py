#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience functions."""

import numpy as np


def iSourcePower(isrc, radius, sigma):
    """
    Get an analytic estimate of the power dissipated by a spherical current source.

    Parameters
    ----------
    isrc : float
        Amplitude of source in amps.
    radius : float
        Radius of source in meters.
    sigma : float or float[:]
        Conductivity of surroundings in S/m.

    Returns
    -------
    power : float
        Estimated power dissipation in watts.

    """
    if type(sigma) == np.ndarray:
        sigma = np.linalg.norm(sigma)

    rInf = 1/(4*np.pi*sigma*radius)
    power = rInf*isrc**2
    return power


def vSourceIntegral(vsrc, radius, rmax):
    """
    Calculate exact integral of V for a spherical voltage source
    from the center to a distance of rmax.

    Parameters
    ----------
    vsrc : float
        Amplitude of source in volts.
    radius : float
        Radius of source in meters.
    rmax : float
        Furthest distance of domain.

    Returns
    -------
    float
        Integral of V from center to rmax.

    """
    return radius*vsrc*(1+np.log(rmax/radius))


def iSourceIntegral(isrc, radius, rmax, sigma):
    """
    Calculate exact integral of V for a spherical current source
    from the center to a distance of rmax.

    Parameters
    ----------
    vsrc : float
        Amplitude of source in volts.
    radius : float
        Radius of source in meters.
    rmax : float
        Furthest distance of domain.
    sigma : float or float[:]
        Conductivity of region in S/m

    Returns
    -------
    float
        Integral of V from center to rmax.

    """
    vsrc = isrc/(4*np.pi*sigma)
    return vSourceIntegral(vsrc, radius, rmax)


def estimatePower(voltages, edges, conductances):
    """
    Estimate power dissipated in the equivalent netowrk of the mesh.

    Parameters
    ----------
    voltages : float[:]
        Voltage at each node.
    edges : int[:,2]
        i-th lists indices of the endpoints for the i-th conductance.
    conductances : float[:]
        Value of conductances in S.

    Returns
    -------
    float
        Total power dissipated in watts.

    """
    dv = voltages[edges]
    v2 = np.diff(dv, axis=1)**2
    return np.dot(v2.squeeze(), conductances)


def getErrorEstimates(simulation):
    """
    Get error-related metrics for a simulation of point/sphere sources.

    Parameters
    ----------
    simulation : `~xcell.Simulation`
        Simulation object.

    Returns
    -------
    data : dict
        Error-related quantitites.

    """
    errSummary, err, vAna, sorter, r = simulation.calculateErrors()
    intErr = np.trapz(err[sorter], r[sorter])

    v = vAna-err

    sse = np.sum(err**2)
    sstot = np.sum((vAna-np.mean(vAna))**2)
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
    """
    Get sums-of-squares for statistical operations

    Parameters
    ----------
    err : float[:]
        Difference between simulated and analytic voltage at each node.
    vAna : float[:]
        Analytic voltage at each node.

    Returns
    -------
    SSE : float
        Sum of squared errors.
    SSTot : float
        Total sum of squares.

    """
    SSE = np.sum((err)**2)
    SSTot = np.sum((vAna-np.mean(vAna))**2)

    return SSE, SSTot


def transposeDicts(dictList):
    """
    Convert list of dicts (from timesteps) to dict of lists (for plotting).

    Parameters
    ----------
    dictList : [{}]
        List of data from each timestep.

    Returns
    -------
    arrayDict : {[]}
        Dict of variables, as lists of values at each timestep.

    """
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
    """
    Calculate fraction of variance unexplained (FVU)

    Parameters
    ----------
    analyticVals : float[:]
        Analytic voltage at each node.
    err : float[:]
        Difference between simulated and analytic voltage at each node.

    Returns
    -------
    FVU : float
        Fraction of variance unexplained.

    """
    # err=simVals-analyticVals

    SSE = np.sum(err**2)
    SSTot = np.sum((analyticVals-np.mean(analyticVals))**2)

    FVU = SSE/SSTot

    return FVU
