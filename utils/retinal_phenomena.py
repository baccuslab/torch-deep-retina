import matplotlib.pyplot as plt
import numpy as np 
from itertools import repeat
import utils.stimuli as stim
import utils.visualizations as viz
from tqdm import tqdm, trange
import torch

DEVICE = torch.device("cuda:0")

def step_response(model, duration=100, delay=50, nsamples=200, intensity=-1.):
    """Generates step responses (using utils.stimuli.flash)

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

    Returns
    -------
    figs : matplotlib.figure.Figure
        Matplotlib Figure object into which step response is plotted    	
    X : array_like of shape (nsamples - 40, 40, 50, 50)
    resp : torch.tensor of size (nsamples - 40, number of cells)
    """
    X = stim.concat(stim.flash(duration, delay, nsamples, intensity=intensity))
    X_torch = torch.from_numpy(X).to(DEVICE)
    resp = model(X_torch)
    figs = viz.response1D(X[:, -1, 0, 0].copy(), resp.cpu().detach().numpy())
    return figs, X, resp


def paired_flash(model, ifis=(2, 20), duration=1, intensity=-2.0, total=100, delay=40):
    """Generates responses to a pair of neighboring flashes
    Parameters
    ----------
    ifi : int
        inter-flash interval, in samples (default: 5)
    duration : int
        the duration of each flash in frames (default: 1)
    intensity : float
        the flash intensity (default: -2.0)
    padding : int
        how much padding in frames to put on either side of the flash (default: 50)
    """
    s1, r1, s2, r2 = [], [], [], []
    stimuli = []
    responses = []

    for ifi in tqdm(np.arange(ifis[0], ifis[1], duration)):
        # single flashes
        x1 = stim.paired_flashes(ifi, duration, (intensity, 0), total, delay)
        s1.append(stim.unroll(x1)[:, 0, 0])
        x1_torch = torch.from_numpy(x1).to(DEVICE)
        r1.append(stim.prepad(model(x1_torch).cpu().detach().numpy()))

        x2 = stim.paired_flashes(ifi, duration, (0, intensity), total, delay)
        x2_torch = torch.from_numpy(x2).to(DEVICE)
        s2.append(stim.unroll(x2)[:, 0, 0])
        r2.append(stim.prepad(model(x2_torch).cpu().detach().numpy()))

        # pair
        x = stim.paired_flashes(ifi, duration, intensity, total, delay)
        x_torch = torch.from_numpy(x).to(DEVICE)
        stimuli.append(stim.unroll(x)[:, 0, 0])
        responses.append(stim.prepad(model(x).cpu().detach().numpy()))

    return map(np.stack, (s1, r1, s2, r2, stimuli, responses))


def reversing_grating(model, size=5, phase=0.):
    """A reversing grating stimulus"""
    grating = stim.grating(barsize=(size, 0), phase=(phase, 0.0), intensity=(1.0, 1.0), us_factor=1, blur=0)
    X = stim.concat(stim.reverse(grating, halfperiod=50, nsamples=300))
    X_torch = torch.from_numpy(X).to(DEVICE)
    resp = model(X_torch)
    figs = viz.response1D(X[:, -1, 0, 0].copy(), resp.cpu().detach().numpy())
    return figs, X, resp


def contrast_adaptation(model, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = np.stack([model(
        torch.from_numpy(stim.concat(np.random.randn(*envelope.shape) * envelope)).to(DEVICE)).cpu().detach().numpy()
                          for _ in trange(nrepeats)])

    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))

    return figs, envelope, responses


def oms(duration=4, sample_rate=0.01, transition_duration=0.07, silent_duration=0.93,
        magnitude=5, space=(50, 50), center=(25, 25), object_radius=5, coherent=False, roll=False):
    """
    Object motion sensitivity stimulus, where an object moves differentially
    from the background.
    INPUT:
    duration        movie duration in seconds
    sample_rate     sample rate of movie in Hz
    coherent        are object and background moving coherently?
    space           spatial dimensions
    center          location of object center
    object_width    width in pixels of object
    speed           speed of random drift
    motion_type     'periodic' or 'drift'
    roll            whether to roll_axis for model prediction
    OUTPUT:
    movie           a numpy array of the stimulus
    """
    # fixed params
    contrast = 1
    grating_width = 3

    transition_frames = int(transition_duration / sample_rate)
    silent_frames = int(silent_duration / sample_rate)
    total_frames = int(duration / sample_rate)

    # silence, one direction, silence, opposite direction
    obj_position = np.hstack([np.zeros((silent_frames,)), np.linspace(0, magnitude, transition_frames),
                              magnitude * np.ones((silent_frames,)), np.linspace(magnitude, 0, transition_frames)]).astype('int')

    half_silent = silent_frames // 2
    back_position = np.hstack([obj_position[half_silent:], obj_position[:-half_silent]]).astype('int')

    # make position sequence last total_frames
    if len(back_position) > total_frames:
        print("Warning: movie won't be {} shorter than a full period.".format(np.float(2 * transition_frames + 2 * silent_frames) / total_frames))
        back_position[:total_frames]
        obj_position[:total_frames]
    else:
        reps = int(np.ceil(np.float(total_frames) / len(back_position)))
        back_position = np.tile(back_position, reps)[:total_frames]
        obj_position = np.tile(obj_position, reps)[:total_frames]

    # create a larger fixed world of bars that we'll just crop from later
    padding = 2 * grating_width + magnitude
    fixed_world = -1 * np.ones((space[0], space[1] + padding))
    for i in range(grating_width):
        fixed_world[:, i::2 * grating_width] = 1

    # make movie
    movie = np.zeros((total_frames, space[0], space[1]))
    for frame in range(total_frames):
        # make background grating
        background_frame = np.copy(fixed_world[:, back_position[frame]:back_position[frame] + space[0]])

        if not coherent:
            # make object frame
            object_frame = np.copy(fixed_world[:, obj_position[frame]:obj_position[frame] + space[0]])

            # set center of background frame to object
            object_mask = stim.cmask(center, object_radius, object_frame)
            background_frame[object_mask] = object_frame[object_mask]

        # adjust contrast
        background_frame *= contrast
        movie[frame] = background_frame
    return movie


def osr(model, duration=2, interval=10, nflashes=3, intensity=-2.0):
    """Omitted stimulus response
    Parameters
    ----------
    duration : int
        Length of each flash in samples (default: 2)
    interval : int
        Number of frames between each flash (default: 10)
    nflashes : int
        Number of flashes to show before the omitted flash (default: 5)
    intensity : float
        The intensity (luminance) of each flash (default: -2.0)
    """

    # generate the stimulus
    single_flash = stim.flash(duration, interval, interval * 2, intensity=intensity)
    omitted_flash = stim.flash(duration, interval, interval * 2, intensity=0.0)
    flash_group = list(repeat(single_flash, nflashes))
    zero_pad = np.zeros((interval, 1, 1))
    X = stim.concat(zero_pad, *flash_group, omitted_flash, *flash_group, nx=50, nh=40)
    X_torch = torch.from_numpy(X).to(DEVICE)
    resp = model(X_torch)
    figs = viz.response1D(X[:, -1, 0, 0].copy(), resp.cpu().detach().numpy(), figsize=(20, 8))
    return figs, X, resp


def motion_anticipation(model, scale_factor=55, velocity=0.08, width=2, flash_duration=2):
    """Generates the Berry motion anticipation stimulus
    Stimulus from the paper:
    Anticipation of moving stimuli by the retina,
    M. Berry, I. Brivanlou, T. Jordan and M. Meister, Nature 1999
    Parameters
    ----------
    model : keras.Model
    scale_factor = 55       # microns per bar
    velocity = 0.08         # 0.08 bars/frame == 0.44mm/s, same as Berry et. al.
    width = 2               # 2 bars == 110 microns, Berry et. al. used 133 microns
    flash_duration = 2      # 2 frames == 20 ms, Berry et. al. used 15ms
    Returns
    -------
    motion : array_like
    flashes : array_like
    """
    # moving bar stimulus and responses
    c_right, speed_right, stim_right = stim.driftingbar(velocity, width)
    resp_right = model(torch.from_numpy(stim_right).to(DEVICE)).cpu().detach().numpy()

    c_left, speed_left, stim_left = stim.driftingbar(-velocity, width, x=(30, -30))
    resp_left = model(torch.from_numpy(stim_left).to(DEVICE)).cpu().detach().numpy()

    # flashed bar stimulus
    flash_centers = np.arange(-25, 26)
    flashes = (stim.flash(flash_duration, 43, 70, intensity=stim.bar((x, 0), width, 50))
               for x in flash_centers)

    # flash responses are a 3-D array with dimensions (centers, stimulus time, cell)
    flash_responses = np.stack([model(torch.from_numpy(stim.concat(f)).to(DEVICE)).cpu().detach().numpy() for f in tqdm(flashes)])

    # pick off the flash responses at a particular time point (the time of the max response)
    max_resp_idx = flash_responses.mean(axis=-1).mean(axis=0).argmax()
    resp_flash = flash_responses[:, max_resp_idx, :]

    # average the response from multiple cells
    avg_resp_right = resp_right.mean(axis=-1)
    avg_resp_left = resp_left.mean(axis=-1)
    avg_resp_flash = resp_flash.mean(axis=-1)

    # normalize the average responses (to plot on the same scale)
    avg_resp_right /= avg_resp_right.max()
    avg_resp_left /= avg_resp_left.max()
    avg_resp_flash /= avg_resp_flash.max()

    # generate the figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(scale_factor * c_left[40:], avg_resp_left, 'g-', label='Left motion')
    ax.plot(scale_factor * c_right[40:], avg_resp_right, 'b-', label='Right motion')
    ax.plot(scale_factor * flash_centers, avg_resp_flash, 'r-', label='Flash')
    ax.legend(frameon=True, fancybox=True, fontsize=18)
    ax.set_xlabel('Position ($\mu m$)')
    ax.set_ylabel('Scaled firing rate')
    ax.set_xlim(-735, 135)

    return (fig, ax), (speed_left, speed_right), (c_right, stim_right, resp_right), (c_left, stim_left, resp_left), (flash_centers, flash_responses)
