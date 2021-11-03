import numpy as np
from scipy import signal

def estfr(bspk, time, sigma=0.01):
    """
    Estimate the instantaneous firing rates from binned spike counts.
    Parameters
    ----------
    bspk : array_like
        Array of binned spike counts (e.g. from binspikes)
    time : array_like
        Array of time points corresponding to bins
    sigma : float, optional
        The width of the Gaussian filter, in seconds (Default: 0.01 seconds)
    Returns
    -------
    rates : array_like
        Array of estimated instantaneous firing rate
    """
    # estimate the time resolution
    dt = float(np.mean(np.diff(time)))

    # Construct Gaussian filter, make sure it is normalized
    tau = np.arange(-5 * sigma, 5 * sigma, dt)
    filt = np.exp(-0.5 * (tau / sigma) ** 2)
    filt = filt / np.sum(filt)
    size = int(np.floor(filt.size / 2))

    # Filter  binned spike times
    return signal.fftconvolve(filt, bspk, mode='full')[size:size + time.size] / dt