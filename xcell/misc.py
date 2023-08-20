#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience functions."""

import numpy as np


def current_source_power(current, radius, sigma):
    """
    Get an analytic estimate of the power dissipated by a spherical current source.

    Parameters
    ----------
    current : float
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
    power = rInf*current**2
    return power


def voltage_source_integral(voltage, radius, r_max):
    """
    Calculate exact integral of V for a spherical voltage source
    from the center to a distance of r_max.

    Parameters
    ----------
    voltage : float
        Amplitude of source in volts.
    radius : float
        Radius of source in meters.
    r_max : float
        Furthest distance of domain.

    Returns
    -------
    float
        Integral of V from center to r_max.

    """
    return radius*voltage*(1+np.log(r_max/radius))


def current_source_integral(current, radius, r_max, sigma):
    """
    Calculate exact integral of V for a spherical current source
    from the center to a distance of r_max.

    Parameters
    ----------
    current : float
        Amplitude of source in amperes.
    radius : float
        Radius of source in meters.
    r_max : float
        Furthest distance of domain.
    sigma : float or float[:]
        Conductivity of region in S/m

    Returns
    -------
    float
        Integral of V from center to r_max.

    """
    voltage = current/(4*np.pi*sigma)
    return voltage_source_integral(voltage, radius, r_max)


def estimate_power(voltages, edges, conductances):
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


# TODO: document dict contents
def get_error_estimates(simulation):
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
    errSummary, err, vAna, sorter, r = simulation.calculate_errors()
    intErr = np.trapz(err[sorter], r[sorter])

    v = vAna-err

    sse = np.sum(err**2)
    sstot = np.sum((vAna-np.mean(vAna))**2)
    FVU = sse/sstot

    elV, elAna, elErr, _ = simulation.estimate_volume_error(basic=True)

    volErr = sum(np.abs(elErr))
    volAna = sum(np.abs(elAna))
    vol = volErr/volAna

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
    }

    return data


def get_statistical_squares(err, vAna):
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


def transpose_dicts(dict_list):
    """
    Convert list of dicts (from timesteps) to dict of lists (for plotting).

    Parameters
    ----------
    dict_list : [{}]
        List of data from each timestep.

    Returns
    -------
    list_dict : {[]}
        Dict of variables, as lists of values at each timestep.

    """
    list_dict = {}

    for d in dict_list:
        emptyDict = len(list_dict.keys()) == 0
        for k, v in zip(d.keys(), d.values()):
            if emptyDict:
                list_dict[k] = [v]
            else:
                list_dict[k].append(v)

    return list_dict


def calculate_fvu(analytic_values, err):
    """
    Calculate fraction of variance unexplained (FVU)

    Parameters
    ----------
    analytic_values : float[:]
        Analytic voltage at each node.
    err : float[:]
        Difference between simulated and analytic voltage at each node.

    Returns
    -------
    FVU : float
        Fraction of variance unexplained.

    """
    SSE, SSTot = get_statistical_squares(err, analytic_values)

    FVU = SSE/SSTot

    return FVU
