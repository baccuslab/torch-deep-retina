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
import skimage.draw
import cv2

from deepretina.experiments import rolling_window

__all__ = ['concat', 'white', 'contrast_steps', 'flash', 'spatialize', 'bar',
           'driftingbar', 'cmask', 'paired_flashes']

def extend_to_edge(img_shape, pt1, pt2):
    """
    Finds the points that cross at the x=0 and x=img_shape[0] along the line
    defined by the shortest distance between the two argued points.
    (Assists in drawing a line across the whole image using cv2)

    img_shape: tuple of ints with len 2
    pt1: tuple of ints with len 2
    pt2: tuple of ints with len 2
    """
    m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
    b = pt1[1]-m*pt1[0]
    new_pt1 = (0,int(b))
    y = m*img_shape[1]+b
    new_pt2 = (img_shape[1], int(y))
    return new_pt1, new_pt2

def center_line_coords(img_shape, angle=0):
    """
    Finds two points that define a line across the center of the image
    with a slope defined by the angle from the rightward horizontal.
    (These points can be used with the cv2.line fxn to draw a line
    with the desired slope crossing through the center of the image)

    img_shape: tuple of ints with len 2
    angle: float or int
        defines the slope of the virtual line in degrees
    """
    img_center = np.array([img_shape[0]//2, img_shape[1]//2])
    # Horizontal
    if angle % 180 == 0:
        coords = [(0, img_center[0]), (img_shape[1], img_center[0])]
    # Vertical
    elif angle % 180 == 90:
        coords = [(img_center[1], 0), (img_center[1], img_shape[0])]
    else:
        rads = (angle-90)*np.pi/180
        vec = np.array([np.sin(rads), np.cos(rads)])
        pt1, pt2 = extend_to_edge(img_shape, np.flip(img_center), np.flip(img_center)+vec)
        coords = [pt1, pt2]
    return coords

def line_thru_center(img, bar_size, angle=0):
    """
    Draws a line through the center of the image with a slope defined
    by the angle and a line width defined by the barsize.

    img: numpy ndarray with shape (H,W)
        the image to have a line drawn on it
    bar_size: int or float
        defines the width of the line
    angle: int or float
        defines the slope of the line in degrees
    """
    coords = center_line_coords(img.shape, angle)
    cv2.line(img, coords[0], coords[1], 1, bar_size)
    return img

def stripes(img_shape, bar_size, angle=0):
    """
    Returns a single frame of white, black alternating stripes angled
    from the rightward horizontal.

    Inputs:
        img_shape: sequence of ints with length 2
            the size of the image to be returned
        bar_size: int
            the size of the stripes in terms of pixels
        angle: float
            the stripes are drawn according to the specified
            angle on the unit circle
    """
    assert img_shape[0] >= 2*bar_size and img_shape[1] >= 2*bar_size
    angle = angle % 360
    if angle % 180 == 0:
        img = np.zeros((bar_size*2, img_shape[1]))
        img[:bar_size] = 1
        img = np.tile(img, (img_shape[0]//(bar_size*2), 1))
        if img.shape[0] < img_shape[0]:
            img = np.concatenate([img, np.ones((img_shape[0]-img.shape[0], img_shape[1]))])
        return img
    elif angle % 90 == 0 and angle % 180 == 90:
        img = np.zeros((bar_size*2, img_shape[0]))
        img[:bar_size] = 1
        img = np.tile(img, (img_shape[1]//(bar_size*2), 1))
        if img.shape[0] < img_shape[1]:
            img = np.concatenate([img, np.ones((img_shape[1]-img.shape[0], img_shape[0]))])
        return img.T
    assert bar_size >= 4 # Sorry, no inherent reason why, just easier to code
    assert img_shape[0] >= 2*bar_size and img_shape[1] >= 2*bar_size
    coords = center_line_coords(img_shape, angle)
    temp_angle = 90-(angle%180)
    y_step = round(bar_size*np.sin(temp_angle*np.pi/180))
    x_step = round(bar_size*np.cos(temp_angle*np.pi/180))
    n_steps_half_x = abs((img_shape[1]//x_step)//2)
    n_steps_half_y = abs((img_shape[0]//y_step)//2)
    n_steps_half = max(n_steps_half_x, n_steps_half_y)
    start_xs = [-x_step*n_steps_half+c[0] for c in coords]
    start_ys = [-y_step*n_steps_half+c[1] for c in coords]
    img = np.zeros(img_shape)
    for i in range(0,int(2*n_steps_half)+1,2):
        new_xs = [int(x+i*x_step) for x in start_xs]
        new_ys = [int(y+i*y_step) for y in start_ys]
        pt1, pt2 = extend_to_edge(img_shape, (new_xs[0], new_ys[0]), (new_xs[1], new_ys[1]))
        cv2.line(img, pt1, pt2, 1, bar_size)
    return img

def motion_reversal(img_shape, start_pt=(0,0), horz_vel=0.5, vert_vel=0, angle=90, 
                                            bar_size=4, n_frames=None, rev_pt=None):
    """

    """

    if rev_pt is None:
        rev_pt = (img_shape[0]//2, img_shape[1]//2)
    img = np.zeros(img_shape)
    pt1, pt2 = center_line_coords(img_shape, angle)
    cv2.line(img, pt1, pt2, 1, bar_size)

    top_pt = pt1 if pt1[1] < pt2[1] else pt2
    if horz_vel != 0:
        n_shifts = top_pt[0]-start_pt[0]
        start_img = np.roll(img, n_shifts, axis=1)
    if vert_vel != 0:
        n_shifts = top_pt[1]-start_pt[1]
        start_img = np.roll(start_img, n_shifts, axis=0)

    if n_frames is None:
        n_frames = abs(int((rev_pt[0]-start_pt[0])/horz_vel*2))

    frames = []
    cumu_horz = 0
    cumu_vert = 0
    horz_flip = False
    vert_flip = False
    for i in range(n_frames):
        cumu_horz += horz_vel
        new_img = np.roll(start_img, int(cumu_horz), axis=1)
        cumu_vert += vert_vel
        new_img = np.roll(new_img, int(cumu_vert), axis=0)
        new_pt = ((start_pt[0]+cumu_horz)%img_shape[0], (start_pt[1]+cumu_vert)%img_shape[1])
        if horz_vel*new_pt[0] >= horz_vel*rev_pt[0] and not horz_flip:
            horz_vel = -horz_vel
            horz_flip = True
        if vert_vel*new_pt[1] >= vert_vel*rev_pt[1] and not vert_flip:
            vert_vel = -vert_vel
            vert_flip = True
        frames.append(new_img)
    return np.asarray(frames)

def circle_mask(center, radius, mask_shape=(50,50)):
    """
    creates a binary mask in the shape of a circle. 1s are inside the circle, 0s are outside.

    center: sequence of ints with len 2
        the coordinates of the center of the circle
    radius: float
        the radius of the circle
    mask_shape: sequence of ints len 2
        the shape of the mask
    """
    row_coords, col_coords = skimage.draw.circle(*center, radius, mask_shape)
    mask = np.zeros(mask_shape)
    mask[row_coords, col_coords] = 1
    return mask
        
def paste_circle(foreground, background, radius, center):
    """
    Pastes the foreground pattern onto the background pattern in a window the shape of a circle.

    foreground: ndarray with shape (H, W)
        the pattern that the circle will take
    backgroun: ndarray with shape (H,W)
        the pattern of the background
    radius: float
        the radius of the circle in pixels
    center: sequence of ints with length 2
        the center coordinates of the circle (0,0 is the upperleftmost pixel)
    """
    row_coords, col_coords = skimage.draw.circle(*center, radius, background.shape)
    img = background.copy()
    img[row_coords, col_coords] = foreground[row_coords, col_coords]
    return img

def random_differential_circle(n_frames=100, bar_size=4, foreground_velocity=0.5, sync_jitters=False, 
                        background_velocity=0.4, image_shape=(50,50), center=(25,25), radius=5,
                        horizontal_foreground=False, horizontal_background=False):
    """
    Creates circle window onto a grating with a background grating.
    The foreground and background gratings jitter at different rates.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_size: int
            the size of the grating bars in pixels
        sync_jitters: bool
            True: movements between background and foreground are synchronized (uses background velocity for jitters)
            False: movements between background and foreground are asynchronous
        foreground_velocity: float
            the foreground (grating circle) movement intensity.
            units of pixels per frame
        background_velocity: float
            the background (grating) movement intensity.
            units of pixels per frame
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical

    Returns:
        frame sequence of shape (n_frames, image_shape[0], image_shape[1])

    """
    bar_size=int(bar_size)
    angle = 0 if horizontal_foreground else 90
    circle_grating = stripes(image_shape, bar_size, angle=angle)
    angle = 0 if horizontal_background else 90
    background_grating = stripes(image_shape, bar_size, angle=angle)
    background_steps = np.random.randint(-1,2,n_frames)
    if sync_jitters:
        foreground_steps = background_steps 
        foreground_velocity = background_velocity
    else:
        foreground_steps = np.random.randint(-1,2,n_frames)
    row_coords, col_coords = skimage.draw.circle(*center, radius, image_shape)
    frames = []
    for i in range(n_frames):
        step_idx = int(background_velocity*i)%n_frames
        if background_velocity > 0:
            background_grating = np.roll(background_grating, background_steps[step_idx], axis=1)
            if background_velocity < 1 and not sync_jitters:
                background_steps[step_idx] = 0
        new_frame = background_grating.copy()
            
        step_idx = int(foreground_velocity*i)%n_frames
        if foreground_velocity > 0:
            circle_grating = np.roll(circle_grating, foreground_steps[step_idx], axis=1)
            if foreground_velocity < 1:
                foreground_steps[step_idx] = 0
        new_frame[row_coords, col_coords] = circle_grating[row_coords, col_coords]
        frames.append(new_frame)
    return np.asarray(frames)

def differential_circle(n_frames=100, bar_size=4, foreground_velocity=0.5, background_velocity=0, 
                        image_shape=(50,50), center=(25,25), radius=5, init_offset=0,
                        horizontal_foreground=False, horizontal_background=False):
    """
    Creates circle window onto a grating that has stripes perpendicular to the background.
    The grating behind this window then rolls differently than the background grating.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_size: int
            the size of the grating bars in pixels
        foreground_velocity: float
            the foreground (grating circle) moves to the right with positive velocities and left with negative.
            units of pixels per frame
        background_velocity: float
            the background (grating) moves to the right with positive velocities and left with negative.
            units of pixels per frame
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        init_offset: int
            the initial offset of the foreground grating from the background grating
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical

    Returns:
        frame sequence of shape (n_frames, image_shape[0], image_shape[1])

    """
    bar_size=int(bar_size)
    angle = 0 if horizontal_foreground else 90
    circle_grating = stripes(image_shape, bar_size, angle=angle)
    circle_grating = np.roll(circle_grating, init_offset, axis=1)
    angle = 0 if horizontal_background else 90
    background_grating = stripes(image_shape, bar_size, angle=angle)
    row_coords, col_coords = skimage.draw.circle(*center, radius, image_shape)
    #mask = circle_mask(center, radius, image_shape)
    frames = []
    for i in range(n_frames):
        background_dist = int(background_velocity*i)
        if background_dist != 0:
            new_frame = np.roll(background_grating, background_dist, axis=1)
        else:
            new_frame = background_grating.copy()
            
        foreground_dist = int(foreground_velocity*i)
        if foreground_dist != 0:
            new_circ = np.roll(circle_grating, foreground_dist, axis=1)
        else:
            new_circ = circle_grating.copy()
        new_frame[row_coords, col_coords] = new_circ[row_coords, col_coords]
        frames.append(new_frame)
    return np.asarray(frames)

def jittered_circle(n_frames=100, bar_size=4, foreground_jitter=0.5, background_jitter=0, 
                        image_shape=(50,50), center=(25,25), radius=5, 
                        horizontal_foreground=False, horizontal_background=False):
    """
    Creates circle window onto a grating that has stripes perpendicular to the background.
    This window then jitters differently than the background grating.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_size: int
            the size of the grating bars in pixels
        foreground_jitter: positive float
            the foreground (grating circle) jitter intensity.
            arbitrary units
        background_jitter: positive float
            the background (grating) jitter intensity.
            arbitrary units
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        horizontal_foreground: bool
            True: the stripes of the foreground are horizontal
            False: the stripes of the foreground are vertical
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical

    Returns:
        frame sequence of shape (n_frames, image_shape[0], image_shape[1])

    """
    bar_size = int(bar_size)
    foreground_jitter = abs(foreground_jitter)
    background_jitter = abs(background_jitter)
    angle = 0 if horizontal_foreground else 90
    circle_grating = stripes(image_shape, bar_size, angle=angle)
    circle_grating = np.roll(circle_grating, 2, axis=1) # Roll to start with unaligned gratings
    row_shifts, col_shifts = np.random.randint(-1,2, n_frames), np.random.randint(-1,2,n_frames) # Make random center shifts
    row_coords, col_coords = skimage.draw.circle(*center, radius, image_shape)

    angle = 0 if horizontal_background else 90
    background_grating = stripes(image_shape, bar_size, angle=angle)
    background_shifts = np.random.randint(-1, 2, n_frames) # Make random background shifts

    frames = []
    for i in range(n_frames):
        shift_idx = int(background_jitter*i)
        if shift_idx != 0:
            background_grating = np.roll(background_grating, background_shifts[shift_idx], axis=1)
            background_shifts[shift_idx] = 0
        new_frame = background_grating.copy()
            
        shift_idx = int(foreground_jitter*i)
        if shift_idx != 0:
            center = (center[0] + row_shifts[shift_idx], center[1] + col_shifts[shift_idx])
            row_coords, col_coords = skimage.draw.circle(*center, radius, image_shape)
            row_shifts[shift_idx], col_shifts[shift_idx] = 0, 0
        new_frame[row_coords, col_coords] = circle_grating[row_coords, col_coords]
        frames.append(new_frame)
    return np.asarray(frames)

def moving_circle(n_frames=100, bar_size=4, foreground_velocity=0.5, background_velocity=0, 
                        image_shape=(50,50), center=(25,25), radius=5, 
                        horizontal_background=False):
    """
    Creates circle window onto a grating that has stripes perpendicular to the background.
    This window then translates differently than the background grating.

    Inputs:
        n_frames: float
            the total number of frames in the movie
        bar_size: int
            the size of the grating bars in pixels
        foreground_velocity: float
            the foreground (grating circle) moves to the right with positive velocities and left with negative.
            units of pixels per frame
        background_velocity: float
            the background (grating) moves to the right with positive velocities and left with negative.
            units of pixels per frame
        image_shape: sequence of ints with length 2
            the image height and width
        center: sequence of ints with length 2
            the center coordinates of the foreground circle. 0,0 is the upper leftmost part of the image.
        radius: float
            the radius of the forground circle
        horizontal_background: bool
            True: the stripes of the background are horizontal
            False: the stripes of the background are vertical

    Returns:
        frame sequence of shape (n_frames, image_shape[0], image_shape[1])

    """
    bar_size=int(bar_size)
    angle = 90 if horizontal_background else 0
    circle_grating = stripes(image_shape, bar_size, angle=angle)
    angle = (angle+90)%180
    background_grating = stripes(image_shape, bar_size, angle=angle)
    row_coords, col_coords = skimage.draw.circle(*center, radius, image_shape)
    #mask = circle_mask(center, radius, image_shape)
    frames = []
    for i in range(n_frames):
        background_dist = int(background_velocity*i)
        if background_dist != 0:
            new_frame = shift(background_grating, background_fill=0, shift=(0,background_dist))
        else:
            new_frame = background_grating.copy()
            
        foreground_dist = int(foreground_velocity*i)
        if foreground_dist != 0:
            new_center = (center[0], center[1]+foreground_dist)
            new_row_coords, new_col_coords = skimage.draw.circle(*new_center, radius, image_shape)
        else:
            new_row_coords, new_col_coords = row_coords, col_coords
        new_frame[new_row_coords, new_col_coords] = circle_grating[new_row_coords, new_col_coords]
        frames.append(new_frame)
    return np.asarray(frames)
        
def shift(img, background_fill=0, shift=(0,0)):
    """
    Shifts an image by the specified amount. The remaining null space is filled with the background fill.    

    img - the image to be shifted shape (H,W)
    background_fill - the value to fill the null space following the shift
    shift - tuple of shift amounts. (vertical shift, horizontal shift)
        a positive vertical shift shifts up, negative shifts down
        a positive horizontal shift shifts right, negative shifts left
    """
    # Row coords (vertical shift)
    vert_fill = np.zeros((abs(shift[0]), img.shape[1]))+background_fill
    if shift[0] < 0: # Shift Down
        shifted = np.concatentate([vert_fill, img[0:shift[0]]], axis=0)
    elif shift[0] > 0: # Shift up
        shifted = np.concatentate([img[shift[0]:img.shape[0]], vert_fill], axis=0)
    else:
        shifted = img[:]

    # Col coords (horizontal shift)
    horz_fill = np.zeros((img.shape[0], abs(shift[1])))+background_fill
    if shift[1] < 0: # Shift left
        shifted = np.concatenate([shifted[:,-shift[1]:shifted.shape[1]], horz_fill], axis=1)
    elif shift[1] > 0: # Shift right
        shifted = np.concatenate([horz_fill, shifted[:,0:-shift[1]]], axis=1)
    return shifted


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


def flash(duration, delay, nsamples, intensity=-1.):
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
        img = intensity * np.ones((1, 1, 1))
    else:
        img = intensity.reshape(1, *intensity.shape)

    assert nsamples > (delay + duration), \
        "The total number samples must be greater than the delay + duration"
    sequence = np.zeros((nsamples,))
    sequence[delay:(delay + duration)] = 1.0
    return sequence.reshape(-1, 1, 1) * img


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


def driftingbar(velocity, width, intensity=-1., x=(-30, 30)):
    """Drifting bar

    Usage
    -----
    >>> centers, stim = driftingbar(0.08, 2)

    Parameters
    ----------
    velocity : float
        bar velocity in pixels / frame (if negative, the bar reverses direction)

    width : int
        bar width in pixels

    Returns
    -------
    centers : array_like
        The center positions of the bar at each frame in the stimulus

    stim : array_like
        The spatiotemporal drifting bar movie
    """
    npts = 1 + int((x[1] - x[0]) / velocity)
    centers = np.linspace(x[0], x[1], npts)
    return centers, velocity, concat(np.stack(map(lambda x: bar((x, 0), width, np.Inf, us_factor=5, blur=0.), centers)))


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


def grating(barsize=(5, 0), phase=(0., 0.), nx=50, intensity=(1., 1.), us_factor=1, blur=0.):
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
    """
    # generate a square wave along each axis
    x = square(barsize[0], nx * us_factor, phase[0], intensity[0])
    y = square(barsize[1], nx * us_factor, phase[1], intensity[1])

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


def get_grating_movie(grating_width=1, switch_every=10, movie_duration=100, mask=False,
                      intensity=1, phase=0, roll=True):
    '''
        Returns a reversing gratings stimulus.

        INPUT:
            grating_width   the width (in checkers) of each horizontal bar
            switch_every    how often (in frames) you want to reverse grating polarity
            movie_duration  number of frames in stimulus
            mask            either False or a np.array of shape (50,50); masks the gratings (i.e., over the receptive field)
            intensity       what is the contrast of the gratings?
            phase           what is the phase of the gratings?

        OUTPUT:
            full_movies     an np.array of shape (movie_duration, 40, 50, 50)
    '''

    # make grating
    grating_frame = -1 * np.ones((50, 50))
    for i in range(grating_width):
        grating_frame[:, (i + phase)::2 * grating_width] = 1
    if mask:
        grating_frame = grating_frame * mask * intensity
    else:
        grating_frame = grating_frame * intensity

    # make movie
    grating_movie = np.zeros((movie_duration, 50, 50))
    polarity_count = 0
    for frame in range(movie_duration):
        polarity_count += 1
        if int(polarity_count / switch_every) % 2 == 0:
            grating_movie[frame] = grating_frame
        else:
            grating_movie[frame] = -1 * grating_frame

    if roll:
        # roll movie axes to get the right shape
        full_movies = rolling_window(grating_movie, 40)
        return full_movies
    else:
        return grating_movie


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
