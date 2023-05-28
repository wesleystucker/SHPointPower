# -*- coding: utf-8 -*-
"""

@author: Wesley S. Tucker
Department of Earth and Environmental Sciences
University of Illinois Chicago
wtucke5@uic.edu
"""

import numpy as np
import pandas as pd
import scipy.stats as sts


def power_corr(clm, slm, glm, hlm, degree=0, confidence_levels=None):
    """
    Finds the correlation coefficient per degree of two sets of
    spherical harmonic coefficients. Data structure is set up if using
    the function spectral_power().
    Default degree is 20. Use degree=l to change, where l is the maximum
    spherical harmonic degree.

    Parameters
    ----------
    clm : float
        Cosine related coefficients of dataset 1
    slm : float
        Sine relate coefficients of dataset 1
    glm : float
        Cosine related coefficients of dataset 2
    hlm : float
        Sine related coefficients of dataset 2
    degree : int, optional
        The maximum degree of spherical harmonic coefficients,
            default is 20
    confidence_levels : list of float, optional
        Confidence levels for the confidence intervals,
            default is [0.8, 0.95, 0.99]

    Returns
    -------
    corr : pd.DataFrame
        DataFrame of correlation coefficients per degree and the confidence
        intervals for specified confidence levels

    """

    if confidence_levels is None:
        confidence_levels = [0.8, 0.95, 0.99]

    # Create a list of numpy arrays,
    # each array is filled with its index value
    # and has length equal to its index value plus one.
    input_arrays = [np.full(shape=x+1, fill_value=x, dtype=float)
                    for x in np.arange(degree+1)]

    # Flatten the list of arrays to a single series and remove the first item.
    deg = pd.Series(np.hstack(tuple(input_arrays)
                              if isinstance(input_arrays, list)
                              else input_arrays)).drop(labels=0, axis=0)

    # Calculate num, cs_sos, and gh_sos values based on the input arrays.
    num = (clm * glm) + (slm * hlm)
    cs_sos = np.power(clm, 2) + np.power(slm, 2)
    gh_sos = np.power(glm, 2) + np.power(hlm, 2)

    # Create a DataFrame with columns 'deg', 'num', 'cs_sos', and 'gh_sos'.
    # Group the DataFrame by 'deg' column and sum the columns from the group.
    corr = pd.concat([deg, num, cs_sos, gh_sos], axis=1).set_axis(
        ['deg', 'num', 'cs_sos', 'gh_sos'], axis=1)
    corr = corr.groupby(by='deg').sum()

    # Calculate 'r' column
    corr['r'] = (corr['num']
                 / np.sqrt(np.multiply(corr['cs_sos'], corr['gh_sos'])))

    # Replace 'deg' column with a sequence of numbers from 1 to degree + 1.
    corr['deg'] = np.arange(1, degree + 1)

    output_cols = ['deg', 'r']

    # Iterate over each value in the list of confidence levels
    for conf_level in confidence_levels:
        alpha = 1 - conf_level

        # Generate column names for the maximum and minimum columns
        # for this confidence level.
        # For example, if the confidence level is 0.95,
        # max_col will be 'max95', and min_col will be 'min95'
        max_col = f"max{int(conf_level*100)}"
        min_col = f"min{int(conf_level*100)}"

        # Calculate the value for the 'max' column for this confidence level
        # Student's t-distribution percentile point function ppf
        # (i.e., inverse of CDF)
        # adjusted for the  degrees of freedom (corr['deg']*2)
        corr_deg_twice = corr['deg'] * 2
        t_ppf_val = sts.t.ppf(1 - alpha / 2, df=corr_deg_twice)
        denominator = (2 * corr['deg']) + np.power(t_ppf_val, 2)
        corr[max_col] = t_ppf_val * np.sqrt(1 / denominator)

        # The value for the 'min' column is just the negative of the 'max'
        corr[min_col] = corr[max_col] * -1

        # Append the column names to the list of output columns
        output_cols.extend([min_col, max_col])

    corr = pd.concat([corr[col] for col in output_cols], axis=1)

    return corr
