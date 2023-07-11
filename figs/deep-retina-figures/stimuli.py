"""
Generate commonly used visual stimuli

Functions in this module either generate a numpy array
that encodes a particular stimulus (e.g. the `flash` function
generates a full-field flash, or the `contrast_steps` function
generates a sequence of contrast step changes), or they are used
for composing multiple stimulus sequences (concatenating them)
and converting them into a spatiotemporal stimulus (using rolling_window)
that can be fed to a Keras model (for example).
"""

from __future__ import absolute_import, division, print_function

from itertools import repeat
from numbers import Number

import numpy as np
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean

from torchdeepretina.stimuli import rolling_window

__all__ = ['concat', 'white', 'contrast_steps', 'flash', 'spatialize', 'bar',
           'driftingbar', 'cmask', 'paired_flashes']


def unroll(X):
    """Unrolls a toeplitz stimulus"""
    return np.vstack((X[0, :], X[:, -1]))


def prepad(y, n=40, v=0):
    """prepads with value"""
    pad = v * np.ones((n, y.shape[1]))
    return np.vstack((pad, y))


def concat(*args, nx=50, nh=40):
    """Returns a spatiotemporal stimulus that has been transformed using
    rolling_window given a list of stimuli to concatenate

    Parameters
    ----------
    stimuli : iterable
        A list or iterable of stimuli (numpy arrays). The first dimension
        is the sample, which can be different, but the rest of the
        dimensions (spatial dimensions) must be the same

    nh : int, optional
        Number of time steps in the rolling window history (default: 40)

    nx : int, optional
        Number of spatial dimensions (default: 50)
    """
    concatenated = np.vstack(map(lambda s: spatialize(s, nx), args)).astype('float32')
    return rolling_window(concatenated, nh)


def white(nt, nx=1, contrast=1.0):
    """Gaussian white noise with the given contrast

    Parameters
    ----------
    nt : int
        number of temporal samples

    nx : int
        number of spatial dimensions (default: 1)

    contrast : float
        Scalar multiplied by the whole stimulus (default: 1.0)
    """
    return contrast * np.random.randn(nt, nx, nx)


def contrast_steps(contrasts, lengths, nx=1):
    """Returns a random sequence with contrast step changes

    Parameters
    ----------
    contrasts : array_like
        List of the contrasts in the sequence

    lengths : int or array_like
        If an integer is given, each sequence has the same length.
        Otherwise, the given list is used as the lengths for each contrast

    nx : int
        Number of spatial dimensions (default: 1)
    """
    if isinstance(lengths, int):
        lengths = repeat(lengths)

    return np.vstack([white(nt, nx=nx, contrast=sigma)
                      for sigma, nt in zip(contrasts, lengths)])


def spatialize(array, nx):
    """Returns a spatiotemporal version of a full field stimulus

    Given an input array of shape (t, 1, 1), returns a new array with
    shape (t, nx, nx) where each temporal value is copied at each
    spatial location

    Parameters
    ----------
    array : array_like
        The full-field stimulus to spatialize

    nx : int
        The number of desired spatial dimensions (along one edge)
    """
    return np.broadcast_to(array, (array.shape[0], nx, nx))


def flash(duration, delay, nsamples, intensity=-1.,baseline=0):
    """Generates a 1D flash

    Parameters
    ----------
    duration : int
        The duration (in samples) of the flash

    delay : int
        The delay (in samples) before the flash starts

    nsamples : int
        The total number of samples in the array

    intensity : float or array_like, optional
        The flash intensity. If a number is given, the flash is a full-field
        flash. Otherwise, if it is a 2D array, then that image is flashed. (default: 1.0)
    """
    # generate the image to flash
    if isinstance(intensity, Number):
        img = np.ones((1, 1, 1))
    else:
        img = np.ones(1, *intensity.shape)

    #assert nsamples > (delay + duration), \
    assert nsamples >= (delay + duration), \
        "The total number samples must be greater than the delay + duration"
    sequence = np.zeros((nsamples,))
    sequence[delay:(delay + duration)] = intensity
    sequence[sequence!=intensity] = baseline
    return sequence.reshape(-1, 1, 1) * img

#def flash(duration, delay, nsamples, intensity=-1.):
#    """Generates a 1D flash
#
#    Parameters
#    ----------
#    duration : int
#        The duration (in samples) of the flash
#
#    delay : int
#        The delay (in samples) before the flash starts
#
#    nsamples : int
#        The total number of samples in the array
#
#    intensity : float or array_like, optional
#        The flash intensity. If a number is given, the flash is a full-field
#        flash. Otherwise, if it is a 2D array, then that image is flashed. (default: 1.0)
#    """
#    # generate the image to flash
#    if isinstance(intensity, Number):
#        img = intensity * np.ones((1, 1, 1))
#    else:
#        img = intensity.reshape(1, *intensity.shape)
#
#    #assert nsamples > (delay + duration), \
#    assert nsamples >= (delay + duration), \
#        "The total number samples must be greater than the delay + duration"
#    sequence = np.zeros((nsamples,))
#    sequence[delay:(delay + duration)] = 1.0
#    return sequence.reshape(-1, 1, 1) * img

def bar(center, width, height, nx=50, intensity=-1., us_factor=1, blur=0.):
    """Generates a single frame of a bar"""

    # upscale factor (for interpolation between discrete bar locations)
    c0 = center[0] * us_factor
    c1 = center[1] * us_factor
    width *= us_factor
    height *= us_factor
    nx *= us_factor

    # center of the bar
    cx, cy = int(c0 + nx // 2), int(c1 + nx // 2)

    # x- and y- indices of the bar
    bx = slice(max(0, cx - width // 2), max(0, min(nx, cx + width // 2)))
    by = slice(max(0, cy - height // 2), max(0, min(nx, cy + height // 2)))

    # set the bar intensity values
    frame = np.zeros((nx, nx))
    frame[by, bx] = intensity

    # downsample the blurred image back to the original size
    return downsample(frame, us_factor, blur)


def downsample(img, factor, blur):
    """Smooth and downsample the image by the given factor"""
    return downscale_local_mean(gaussian(img, blur), (factor, factor))


def driftingbar(speed=0.08, width=2, intensity=-1., x=(-30, 30)):
    """Drifting bar

    Usage
    -----
    >>> centers, stim = driftingbar(0.08, 2)

    Parameters
    ----------
    speed : float
        bar speed in pixels / frame

    width : int
        bar width in pixels

    x : (int, int)
        start and end positions of the bar

    Returns
    -------
    centers : array_like
        The center positions of the bar at each frame in the stimulus

    stim : array_like
        The spatiotemporal drifting bar movie
    """
    npts = int(1 + np.floor(np.abs(x[1] - x[0]) / speed))
    centers = np.linspace(x[0], x[1], npts)

    # convert speed in pixels/frame to mm/s
    dx = 0.055      # mm/pixel
    dt = 0.01       # s/frame
    speed = speed * (dx / dt)

    return centers, speed, np.stack(map(lambda x: bar((x, 0), width, np.Inf, us_factor=5, blur=0.), centers))


def cmask(center, radius, array):
    """Generates a mask covering a central circular region"""
    a, b = center
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    return x ** 2 + y ** 2 <= radius ** 2


def paired_flashes(ifi, duration, intensity, total_length, delay):
    """Paired flash stimulus"""
    # Convert numbers to tuples
    duration = tuplify(duration, 2)
    intensity = tuplify(intensity, 2)

    # generate the flashes
    f0 = flash(duration[0], delay, total_length, intensity[0])
    f1 = flash(duration[1], delay + duration[0] + ifi, total_length, intensity[1])

    # return the concatenated pair
    return concat(f0 + f1)


def square(halfperiod, nsamples, phase=0., intensity=1.0):
    """Generates a 1-D square wave"""
    assert 0 <= phase <= 1, "Phase must be a fraction between 0 and 1"

    # if halfperiod is zero, return all ones
    if halfperiod == 0:
        return np.ones(nsamples)

    # discretize the offset in terms of the period
    offset = int(2 * phase * halfperiod)

    # generate one period of the waveform
    waveform = np.stack(repeat(np.array([intensity, -intensity]), halfperiod)).T.ravel()

    # generate the repeated sequence
    repeats = int(np.ceil(nsamples / (2 * halfperiod)) + 1)
    sequence = np.hstack(repeat(waveform, repeats))

    # use the offset to specify the phase
    return sequence[offset:(nsamples + offset)]


def sinusoid(halfperiod, nsamples, phase=0., intensity=1.0):
    """Generates a 1-D sinusoidal wave"""
    assert 0 <= phase <= 1, "Phase must be a fraction between 0 and 1"

    # if halfperiod is zero, return all ones
    if halfperiod == 0:
        return np.ones(nsamples)

    # compute the base frequency
    time = np.linspace(0, 1, nsamples)
    freq = 1 / (2 * halfperiod / nsamples)

    # convert to radians
    omega = 2 * np.pi * freq
    phi = 2 * np.pi * phase

    return intensity * np.sin(omega * time + phi)


def grating(barsize=(5, 0), phase=(0., 0.), nx=50, intensity=(1., 1.), us_factor=1, blur=0., waveform='square'):
    """Returns a grating as a spatial frame

    Parameters
    ----------
    barsize : (int, int), optional
        Size of the bar in the x- and y- dimensions. A size of 0 indicates no spatial
        variation along that dimension. Default: (5, 0)

    phase : (float, float), optional
        The phase of the grating in the x- and y- dimensions (as a fraction of the period).
        Must be between 0 and 1. Default: (0., 0.)

    intensity=(1., 1.)
        The contrast of the grating for the x- and y- dimensions

    nx : int, optional
        The number of pixels along each dimension of the stimulus (default: 50)

    us_factor : int
        Amount to upsample the image by (before downsampling back to 50x50), (default: 1)

    blur : float
        Amount of blur to applied to the upsampled image (before downsampling), (default: 0.)

    waveform : str
        Either 'square' or 'sin' or 'sinusoid'
    """
    if waveform == 'square':
        wform = square
    elif waveform in ('sin', 'sinusoid'):
        wform = sinusoid
    else:
        raise ValueError(f'Invalid waveform: {waveform}.')

    # generate a square wave along each axis
    x = wform(barsize[0], nx * us_factor, phase[0], intensity[0])
    y = wform(barsize[1], nx * us_factor, phase[1], intensity[1])

    # generate the grating frame and downsample
    return downsample(np.outer(y, x), us_factor, blur)


def jittered_grating(nsamples, sigma=0.1, size=3):
    """Creates a grating that jitters over time according to a random walk"""
    phases = np.cumsum(sigma * np.random.randn(nsamples)) % 1.0
    frames = np.stack([grating(barsize=(size, 0), phase=(p, 0.)) for p in phases])
    return frames


def drifting_grating(nsamples, dt, barsize, us_factor=1, blur=0.):
    """Generates a drifting vertical grating

    Parameters
    ----------
    nsamples : int
        The total number of temporal samples

    dt : float
        The timestep of each sample. A smaller value of dt will generate a slower drift

    barsize : int
        The width of the bar in samples

    us_factor : int, optional
        Amount to upsample the image by (before downsampling back to 50x50), (default: 1)

    blur : float, optional
        Amount of blur to applied to the upsampled image (before downsampling), (default: 0.)
    """
    phases = np.mod(np.arange(nsamples) * dt, 1)
    return np.stack([grating(barsize=(barsize, 0),
                             phase=(phi, 0.),
                             us_factor=us_factor,
                             blur=blur) for phi in phases])


def reverse(img, halfperiod, nsamples):
    """Generates a temporally reversing stimulus using the given image

    Parameters
    ----------
    img : array_like
        A spatial image to reverse (e.g. a grating)

    halfperiod : int
        The number of frames each half period of the reversing image is shown for

    nsamples : int
        The total length of the stimulus in samples
    """
    return np.stack([t * img for t in square(halfperiod, nsamples)])


def motion_reversal(xflip, speed=0.24):
    c1, _, X1 = driftingbar(speed=speed, x=(-40, xflip))
    c2, _, X2 = driftingbar(speed=speed, x=(xflip, -40))
    centers = np.concatenate((c1, c2))[40:]
    tflip = np.where(np.diff(centers) == 0)[0][0]
    stim = concat(X1, X2)
    # resp = model.predict(stim)
    return centers, tflip, stim


def reversing_grating(nsamples, halfperiod, barsize, phase, waveform):
    return reverse(grating(barsize=(barsize, 0), phase=(phase, 0),
                                                waveform=waveform),
                                                halfperiod, nsamples)


def tuplify(x, n):
    """Converts a number into a tuple with that number repeating

    Usage
    -----
    >>> tuplify(3, 5)
    (3, 3, 3, 3, 3)
    >>> tuplify((1,2), 5)
    (1, 2)
    """
    if isinstance(x, Number):
        x = tuple(repeat(x, n))
    return x
