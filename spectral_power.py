# -*- coding: utf-8 -*-
"""
@author: Wesley S. Tucker
Department of Earth and Environmental Sciences,
University of Illinois Chicago
wtucke5@uic.edu
"""


import numpy as np
import pandas as pd
from pyshtools import legendre as leg


def spectral_power(lat, long, coefs=None, pwr_per_deg=None, degree=20):
    """
    Computes spectral power and spherical harmonic coefficients up to a
    maximum degree l and order m of given latitudes and longitudes.

    Parameters
    ----------
    lat : array-like
        Latitudes of points.
    long : array-like
        Longitudes of points.
    coefs : str, optional
        Filename for output of spherical harmonic coefficients as a csv file
            (default is None).
    pwr_per_deg : str, optional
        Filename for output of spectral power per degree as a csv file
            (default is None).
    degree : int, optional
        Maximum spherical harmonic degree to use (default is 20).

    Returns
    -------
    dict
        A dictionary with keys 'power' and 'lmcs' containing the spectral
        power and spherical harmonic coefficients dataframes, respectively.
    """

    colat = 90 - lat
    theta = np.radians(colat)
    phi = np.radians(long)
    n_data = len(lat)

    # Associated Legendre Polynomials
    plm = np.vstack([leg.legendre(degree, x, normalization="ortho",
                    csphase=-1, packed=True) for x in np.cos(theta)]).T

    # Degree l and order m
    degrees = np.arange(degree + 1)
    order = np.hstack([np.arange(x + 1) for x in degrees])
    deg = np.hstack([np.full(x + 1, x) for x in degrees]).astype(float)

    # Prepare phi & m for multiplication
    # and matrix of cos(m*phi) and sin(m*phi)
    cos_m_phi = np.cos(np.outer(phi, order)).T
    sin_m_phi = np.sin(np.outer(phi, order)).T

    # Calculate coefficients
    clm = np.sum(plm * cos_m_phi, axis=1)
    slm = np.sum(plm * sin_m_phi, axis=1)

    # Calculate power
    deg_unique = np.unique(deg)
    sum_clm = {d: np.power(clm[deg == d], 2).sum() for d in deg_unique}
    sum_slm = {d: np.power(slm[deg == d], 2).sum() for d in deg_unique}
    pow_const = 4 * np.pi
    pow_denom = n_data * (2 * deg_unique + 1)
    s_power = pow_const / pow_denom * \
        (np.array(list(sum_clm.values())) + np.array(list(sum_slm.values())))
    # Create dataframes
    lmcs = (pd.DataFrame({'deg': deg[1:].astype(
        int), 'ord': order[1:].astype(int), 'clm': clm[1:], 'slm': slm[1:]}))
    power = pd.DataFrame({'Degree': deg_unique[1:], 'Power': s_power[1:]})

    # Save to CSV if specified
    if coefs:
        lmcs.to_csv(coefs, header=None, index=None)
    if pwr_per_deg:
        power.to_csv(pwr_per_deg, header=None, index=None)

    return {'power': power, 'lmcs': lmcs}
