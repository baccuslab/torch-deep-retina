"""
Model visualization tools
"""
import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
import os
from matplotlib import gridspec
from scipy.interpolate import interp1d
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


def despine(ax):
    """Gets rid of the top and right spines"""
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def animate(frames, filename, dt=0.01, fps=24):
    """animation"""

    # time array
    time = np.arange(frames.shape[0]) * dt

    # frames
    x = frames.copy()
    x -= x.min()
    x /= x.max()
    x *= 255
    x = np.round(x).astype('int')

    def make_frame(t):
        idx = int(t / dt)
        return np.atleast_3d(x[idx]) * np.ones((1, 1, 3))

    # write the video to disk
    anim = VideoClip(make_frame, duration=time[-1])
    anim.write_videofile(filename, fps=fps)


def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with the bleeding edge version of moviepy (not the pip version)
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """
    from moviepy.editor import ImageSequenceClip

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # broadcast along the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def block(x, dt=0.01, us_factor=50, ax=None, alpha=1.0, color='lightgrey'):

    # upsample the stimulus
    time = np.arange(x.size) * dt
    time_us = np.linspace(0, time[-1], us_factor * time.size)
    x = interp1d(time, x, kind='nearest')(time_us)
    maxval, minval = x.max(), x.min()

    ax = plt.gca() if ax is None else ax
    ax.fill_between(time_us, np.zeros_like(x), x, color=color, interpolate=True)
    ax.plot(time_us, x, '-', color=color, alpha=alpha)
    ax.set_xlim(0, time_us[-1] + dt)
    ax.set_yticks([minval, maxval])
    ax.set_yticklabels([minval, maxval], fontsize=22)
    adjust_spines(ax, spines=('left'))

    return ax


def response1D(x, r, dt=0.01, us_factor=50, figsize=(8, 5)):
    """Plots a response given a 1D (temporal) representation of the stimulus
    Parameters
    ----------
    x : array_like
        A 1-D representation of the stimulus
    x : array_like
        A (t, n) array of the response of n cells to the t time points in the stimulus
    dt : float
        The temporal sampling period, in seconds (default: 0.01)
    us_factor : int
        How much to upsample the stimulus by before plotting it (used to get a cleaner picture)
    figsize : tuple
        The figure dimensions. (default: (16, 10))
    """
    assert x.size == r.shape[0], "Dimensions do not agree"
    time = np.arange(x.size) * dt
    nrows = 6
    nspan = 5

    fig = plt.figure(figsize=figsize)
    ax0 = plt.subplot2grid((nrows, 1), (0, 0), rowspan=(nrows - nspan))
    ax1 = plt.subplot2grid((nrows, 1), (nrows - nspan, 0), rowspan=nspan)

    # 1D stimulus trace
    block(x, ax=ax0)
    ax0.set_ylim(-1.05, 0.05)

    # neural responses
    ax1.plot(time, r, '-', color='firebrick')
    ax1.set_xlim(0, time[-1] + dt)
    ax1.set_ylim(-0.05, r.max() + 0.05)
    ax1.set_ylabel('Firing rate (Hz)')
    ax1.set_xlabel('Time (s)')
    adjust_spines(ax1)

    return fig, (ax0, ax1)


def plot_traces_grid(weights, tax=None, color='k', lw=3):
    """Plots the given array of 1D traces on a grid
    Parameters
    ----------
    weights : array_like
        Must have shape (num_rows, num_cols, num_dimensions)
    Returns
    -------
    fig : a matplotlib figure handle
    """

    # create the figure
    fig = plt.figure(figsize=(12, 8))

    # number of convolutional filters and number of affine filters
    nrows = weights.shape[0]
    ncols = weights.shape[1]
    ix = 1

    # keep everything on the same y-scale
    ylim = (weights.min(), weights.max())

    # time axis
    if tax is None:
        tax = np.arange(weights.shape[2])

    for row in range(nrows):
        for col in range(ncols):

            # add the subplot
            ax = fig.add_subplot(nrows, ncols, ix)
            ix += 1

            # plot the filter
            ax.plot(tax, weights[row, col], color=color, lw=lw)
            ax.set_ylim(ylim)
            ax.set_xlim((tax[0], tax[-1]))

            if row == col == 0:
                ax.set_xlabel('Time (s)')
            else:
                plt.grid('off')
                plt.axis('off')

    plt.show()
    plt.draw()
    return fig


def plot_filters(weights, cmap='seismic', normalize=True):
    """Plots an array of spatiotemporal filters
    Parameters
    ----------
    weights : array_like
        Must have shape (num_conv_filters, num_temporal, num_spatial, num_spatial)
    cmap : str, optional
        A matplotlib colormap (Default: 'seismic')
    normalize : boolean, optional
        Whether or not to scale the color limit according to the minimum and maximum
        values in the weights array
    Returns
    -------
    fig : a matplotlib figure handle
    """

    # create the figure
    fig = plt.figure(figsize=(12, 8))

    # number of convolutional filters
    num_filters = weights.shape[0]

    # get the number of rows and columns in the grid
    nrows, ncols = gridshape(num_filters, tol=2.0)

    # build the grid for all of the filters
    outer_grid = gridspec.GridSpec(nrows, ncols)

    # normalize to the maximum weight in the array
    if normalize:
        max_val = np.max(abs(weights.ravel()))
        vmin, vmax = -max_val, max_val
    else:
        vmin = np.min(weights.ravel())
        vmax = np.max(weights.ravel())

    # loop over each convolutional filter
    for w, og in zip(weights, outer_grid):

        # get the spatial and temporal frame
        spatial, temporal = ft.decompose(w)

        # build the gridspec (spatial and temporal subplots) for this filter
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=og, height_ratios=(4, 1), hspace=0.0)

        # plot the spatial frame
        ax = plt.Subplot(fig, inner_grid[0])
        ax.imshow(spatial, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.add_subplot(ax)
        plt.grid('off')
        plt.axis('off')

        ax = plt.Subplot(fig, inner_grid[1])
        ax.plot(temporal, 'k', lw=2)
        fig.add_subplot(ax)
        plt.grid('off')
        plt.axis('off')

    plt.show()
    plt.draw()
    return fig


def reshape_affine(weights, num_conv_filters):
    """Reshapes a layer of affine weights (loaded from a keras h5 file)
    Takes a weights array (from a keras affine layer) that has dimenions (S*S*C) x (A), and reshape
    it to be C x A x S x S, where C is the number of conv filters (given), A is the number of affine
    filters, and S is the number of spatial dimensions in each filter
    """
    num_spatial = np.sqrt(weights.shape[0] / num_conv_filters)
    assert np.mod(num_spatial, 1) == 0, 'Spatial dimensions are not square, check the num_conv_filters'

    newshape = (num_conv_filters, int(num_spatial), int(num_spatial), -1)
    return np.rollaxis(weights.reshape(*newshape), -1, 1)


def plot_spatial_grid(weights, cmap='seismic', normalize=True):
    """Plots the given array of spatial weights on a grid
    Parameters
    ----------
    weights : array_like
        Must have shape (num_conv_filters, num_affine_filters, num_spatial, num_spatial)
    cmap : str, optional
        A matplotlib colormap (Default: 'seismic')
    normalize : boolean, optional
        Whether or not to scale the color limit according to the minimum and maximum
        values in the weights array
    Returns
    -------
    fig : a matplotlib figure handle
    """

    # create the figure
    fig = plt.figure(figsize=(16, 12))

    # number of convolutional filters and number of affine filters
    nrows = weights.shape[0]
    ncols = weights.shape[1]
    ix = 1

    # normalize to the maximum weight in the array
    if normalize:
        max_val = np.max(abs(weights.ravel()))
        vmin, vmax = -max_val, max_val
    else:
        vmin = np.min(weights.ravel())
        vmax = np.max(weights.ravel())

    for row in range(nrows):
        for col in range(ncols):

            # add the subplot
            ax = fig.add_subplot(nrows, ncols, ix)
            ix += 1

            # plot the filter
            ax.imshow(weights[row, col], interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.grid('off')
            plt.axis('off')

    plt.show()
    plt.draw()
    return fig


def gridshape(n, tol=2.0):
    """Generates the dimensions of a grid containing n elements
    Parameters
    ----------
    n : int
        The number of elements that need to fit inside the grid
    tol : float
        The maximum allowable aspect ratio in the grid (e.g. a tolerance
        of 2 allows for a grid where one dimension is twice as long as
        the other). (Default: 2.0)
    Examples
    --------
    >>> gridshape(13)
    (3, 5)
    >>> gridshape(8, tol=2.0)
    (2, 4)
    >>> gridshape(8, tol=1.5)
    (3, 3)
    """

    # shortcut if n is small (fits on a single row)
    if n <= 5:
        return (1, n)

    def _largest_fact(n):
        k = np.floor(np.sqrt(n))
        for j in range(int(k), 0, -1):
            if np.mod(n, j) == 0:
                return j, int(n / j)

    for j in np.arange(n, np.ceil(np.sqrt(n)) ** 2 + 1):
        a, b = _largest_fact(j)
        if (b / a <= tol):
            return (a, b)


def adjust_spines(ax, spines=('left', 'bottom')):
    """Modify the spines of a matplotlib axes handle
    Usage
    -----
    >>> adjust_spines(plt.gca(), spines=['left', 'bottom'])
    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def savemov(movies, subplots, filename, cmaps, T=None, clim=None, fps=15, figsize=None):

    # Set up the figure
    fig, axs = plt.subplots(*subplots, figsize=figsize)

    # total length
    if T is None:
        T = movies[0].shape[0]

    # mean subtract
    xs = []
    imgs = []
    for movie, ax, cmap in zip(movies, axs, cmaps):
        X = movie.copy()
        # X -= X.mean()
        xs.append(X)

        img = ax.imshow(X[0])
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(0, X.shape[1])
        ax.set_ylim(0, X.shape[2])
        ax.set_xticks([])
        ax.set_yticks([])

        # Set up the colors
        img.set_cmap(cmap)
        img.set_interpolation('nearest')
        if clim is not None:
            img.set_clim(clim)
        else:
            maxval = np.max(np.abs(X))
            img.set_clim([-maxval, maxval])
        imgs.append(img)

    plt.show()
    plt.draw()

    dt = 1 / fps

    def animate(t):
        for X, img, ax in zip(xs, imgs, axs):
            i = np.mod(int(t / dt), T)
            ax.set_title(f't={i*0.01:2.2f} s')
            img.set_data(X[i])
        return mplfig_to_npimage(fig)

    animation = VideoClip(animate, duration=T * dt)
    # animation.write_gif(filename + '.gif', fps=fps)
    animation.write_videofile(filename + '.mp4', fps=fps)