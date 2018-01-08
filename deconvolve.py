import numpy as np
from sklearn import linear_model


# Convolve and downsample event timecourses to TR timecourses
def regressor_to_TR(E, TR, nTR):
    T = E.shape[0]
    nEvents = E.shape[1]

    # HRF (from AFNI)
    dt = np.arange(0, 15)
    p = 8.6
    q = 0.547
    hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

    # Convolve event matrix to get design matrix
    design_seconds = np.zeros((T, nEvents))
    for e in range(nEvents):
        design_seconds[:, e] = np.convolve(E[:, e], hrf)[:T]

    # Downsample event matrix to TRs
    timepoints = np.linspace(0, (nTR - 1) * TR, nTR)
    design = np.zeros((len(timepoints), nEvents))
    for e in range(nEvents):
        design[:, e] = np.interp(timepoints, np.arange(0, T),
                                 design_seconds[:, e])
        design[:, e] = design[:, e] / np.max(design[:, e])

    return design


# Run linear regression to deconvolve
def deconv(V, design):
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(design, V.T)
    return regr.coef_
