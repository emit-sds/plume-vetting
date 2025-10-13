#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit


def transmittance_model( wavelengths, *args, **kwargs):
    r"""Parameterized discrete plume transmittance model.

    Implements eq. 6 in Xiang, et al., "Identification of false methane plumes
    for orbital imaging spectrometers: A case study with EMIT" (modified here
    for clarification):

    f(\lambda,\left(\boldsymbol\theta,\alpha\right)) =
        \left(\sum\limits_{i=0}^{N}\theta_{i}\lambda^{i}\right) e^{-\epsilon(\lambda)\alpha}

    Args:
        wavelengths (double, vector-like): Wavelength values, lambda, at which
            discrete channel absorption coefficient, epsilon, has been defined.
        *args (double): Variable length argument list consisting of alpha
            (concentration length), theta_i curve fit parameters (in that
            order), where curve fit parameters are ordered as (ref.
            numpy.polyval):
            theta[0]*lambda**(N-1) + theta[1]*lambda**(N-2) + ... + theta[N-2]**lambda + theta[N-1]
        **kwargs (double): epsilon (vector-like) absorption coefficient, as
            function of wavelengths (note: len(epsilon)==len(wavelengths)).

    Returns:
        Discretized, approximate plume transmittance (double) evaluated at
        wavelengths.

    Notes:
        - The transmittance model implemented here assumes that the values
        provided for wavelengths correspond to the discretization basis of
        epsilon (i.e., eps_0 = eps(wavelength_0), eps_1 = eps(wavelength_1),
        etc.)
        - Argument order, convention, has been defined to facilitate this
        function's use with scipy.optimize.curve_fit().

    Raises:
        RuntimeError: If len(wavelengths) is not equal to len(epsilon).

    """
    wavelengths = np.array(wavelengths)
    alpha = args[0]
    theta = args[1:]
    epsilon = np.array(kwargs['epsilon'])

    try:
        len(wavelengths)==len(epsilon)
    except:
        raise RuntimeError('len(wavelengths) must equal len(epsilon)')

    return np.polyval(theta,wavelengths) * np.exp(-epsilon*alpha)

