import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
from pyret.nonlinearities import Binterp, Sigmoid
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import collections
from tqdm import tqdm
from itertools import repeat
import torchdeepretina.stimuli as tdrstim
import torchdeepretina.visualizations as viz
import torchdeepretina.utils as tdrutils
from tqdm import tqdm, trange
import torch

DEVICE = torch.device("cuda:0")
device = DEVICE

def step_response(model, duration=100, delay=50, nsamples=200, intensity=-1., filt_depth=40):
    """Step response"""
    flash = tdrstim.flash(duration, delay, nsamples, intensity=intensity)
    X = tdrstim.concat(flash, nh=filt_depth)
    X_torch = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(X_torch.shape[0]):
                resp, hs = model(X_torch[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(X_torch)
    figs = viz.response1D(X[:, -1, 0, 0].copy(), resp.cpu().detach().numpy())
    (fig, (ax0,ax1)) = figs
    return (fig, (ax0,ax1)), X, resp


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
        x1 = tdrstim.paired_flashes(ifi, duration, (intensity, 0), total, delay)
        s1.append(tdrstim.unroll(x1)[:, 0, 0])
        x1_torch = torch.from_numpy(x1).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x1_torch.shape[0]):
                    resp, hs = model(x1_torch[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x1_torch)
        r1.append(tdrstim.prepad(resp.cpu().detach().numpy()))

        x2 = tdrstim.paired_flashes(ifi, duration, (0, intensity), total, delay)
        x2_torch = torch.from_numpy(x2).to(DEVICE)
        s2.append(tdrstim.unroll(x2)[:, 0, 0])
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x2_torch.shape[0]):
                    resp, hs = model(x2_torch[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x2_torch)
        r2.append(tdrstim.prepad(resp.cpu().detach().numpy()))

        # pair
        x = tdrstim.paired_flashes(ifi, duration, intensity, total, delay)
        x_torch = torch.from_numpy(x).to(DEVICE)
        stimuli.append(tdrstim.unroll(x)[:, 0, 0])
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x_torch.shape[0]):
                    resp, hs = model(x_torch[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x_torch)
        responses.append(tdrstim.prepad(resp.cpu().detach().numpy()))

    return map(np.stack, (s1, r1, s2, r2, stimuli, responses))


def reversing_grating(model, size=5, phase=0., filt_depth=40):
    """A reversing grating stimulus"""
    grating = tdrstim.grating(barsize=(size, 0), phase=(phase, 0.0), intensity=(1.0, 1.0), us_factor=1, blur=0)
    X = tdrstim.concat(tdrstim.reverse(grating, halfperiod=50, nsamples=300), nh=filt_depth)
    X_torch = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(X_torch.shape[0]):
                resp, hs = model(X_torch[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(X_torch)
    figs = viz.response1D(X[:, -1, 0, 0].copy(), resp.cpu().detach().numpy())
    (fig, (ax0,ax1)) = figs
    return (fig, (ax0,ax1)), X, resp


def contrast_adaptation(model, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = tdrstim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    with torch.no_grad():
        for _ in trange(nrepeats):
            x = torch.from_numpy(tdrstim.concat(np.random.randn(*envelope.shape) * envelope, nh=filt_depth)).to(DEVICE)
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
            responses.append(resp.cpu().detach().numpy())

    responses = np.asarray(responses)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return (fig, (ax0,ax1)), envelope, responses

def oms_random_differential(model, duration=5, sample_rate=30, pre_frames=40, post_frames=40, img_shape=(50,50), center=(25,25), radius=8, background_velocity=.3, foreground_velocity=.5, seed=None, bar_size=2, inner_bar_size=None, filt_depth=40):
    """
    Plays a video of differential motion by keeping a circular window fixed in space on a 2d background grating.
    A grating exists behind the circular window that moves counter to the background grating. Each grating is jittered
    randomly.

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_frames: int
        number of frames of still image to be prepended to the jittering
    post_frames: int
        number of frames of still image to be appended to the jittering
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window (0,0 is the upper left most pixel)
    radius: float
        the radius of the circular window
    background_velocity: float
        the intensity of the horizontal jittering of the background grating
    foreground_velocity: float
        the intensity of the horizontal jittering of the foreground grating
    seed: int or None
        sets the numpy random seed if int
    bar_size: int
        size of stripes. Min value is 3
    inner_bar_size: int
        size of grating bars inside circle. If None, set to bar_size
    """
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration * sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    differential, _, _ = tdrstim.random_differential_circle(diff_frames, bar_size=bar_size, inner_bar_size=inner_bar_size,
                                    foreground_velocity=foreground_velocity, 
                                    background_velocity=background_velocity,
                                    image_shape=img_shape, center=center, radius=radius) 
    pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
    post_vid = np.repeat(differential[-1:], post_frames, axis=0)
    diff_vid = np.concatenate([pre_vid, differential, post_vid], axis=0)

    global_velocity = foreground_velocity if foreground_velocity != 0 else background_velocity
    global_, _, _ = tdrstim.random_differential_circle(diff_frames, bar_size=bar_size, inner_bar_size=inner_bar_size,
                                    foreground_velocity=global_velocity, sync_jitters=True,
                                    background_velocity=global_velocity, 
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(global_[:1], pre_frames, axis=0)
    post_vid = np.repeat(global_[-1:], post_frames, axis=0)
    global_vid = np.concatenate([pre_vid, global_, post_vid], axis=0)
    
    if model is None:
        fig = None
        diff_response = None
        global_response = None
    else:
        x = torch.FloatTensor(tdrstim.concat(diff_vid, nh=filt_depth)).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        diff_response = resp.cpu().detach().numpy()

        x = torch.FloatTensor(tdrstim.concat(global_vid, nh=filt_depth)).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        global_response = resp.cpu().detach().numpy()

        # generate the figure
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(diff_response.mean(-1), color="g")
        ax.plot(global_response.mean(-1), color="b")
        ax.legend(["diff", "global"])
        diff_response = diff_response[pre_frames-40:tot_frames-post_frames]
        global_response = global_response[pre_frames-40:tot_frames-post_frames]
    return fig, diff_vid, global_vid, diff_response, global_response

def random_differential(duration=5, sample_rate=30, pre_frames=40, 
                        post_frames=40, img_shape=(50,50), center=(25,25),     
                        radius=8, background_velocity=1, foreground_velocity=1, 
                        seed=None, bar_size=2, n_loops=5):
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration * sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    back_grate = stripes(img_shape, bar_size, angle=0)
    circ_grate = stripes(img_shape, bar_size, angle=0)
    vid = []
    for i in range(n_loops):
        differential, back_grate, circ_grate = tdrstim.random_differential_circle(diff_frames, 
                                                    bar_size=bar_size,
                                                    foreground_velocity=foreground_velocity, 
                                                    background_velocity=background_velocity,
                                                    horizontal_background=True, horizontal_foreground=True,
                                                    image_shape=img_shape, center=center, radius=radius,
                                                    background_grating=back_grate, circle_grating=circ_grate)
        post_vid = np.tile(back_grate[None], (post_frames,1,1))
        if i == 0:
            pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
            diff_vid = np.concatenate([pre_vid, differential, post_vid], axis=0)            
        else:
            diff_vid = np.concatenate([differential, post_vid], axis=0)
        vid.append(diff_vid)

        global_velocity = foreground_velocity if foreground_velocity != 0 else background_velocity
        global_, back_grate, circ_grate = tdrstim.random_differential_circle(diff_frames, bar_size=bar_size, 
                                        foreground_velocity=global_velocity, sync_jitters=True,
                                        background_velocity=global_velocity, 
                                        image_shape=img_shape, center=center, radius=radius, 
                                        horizontal_foreground=True, horizontal_background=True,
                                        background_grating=back_grate, circle_grating=back_grate.copy())
        post_vid = np.repeat(global_[-1:], post_frames, axis=0)
        global_vid = np.concatenate([global_, post_vid], axis=0)
        vid.append(global_vid)
    return np.concatenate(vid, axis=0)

def periodic_differential(duration=5, sample_rate=30, pre_frames=40, 
                        post_frames=40, img_shape=(50,50), center=(25,25),     
                        radius=8, period_dur=30, bar_size=2, n_loops=5, n_steps=3):
    tot_frames = int(duration * sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    back_grate = stripes(img_shape, bar_size, angle=0)
    circ_grate = stripes(img_shape, bar_size, angle=0)
    vid = []
    for i in range(n_loops):
        differential, back_grate, circ_grate = tdrstim.periodic_differential_circle(n_frames=diff_frames, 
                                                    period_dur=30, sync_periods=False, image_shape=img_shape,
                                                    center=center, radius=radius, horizontal_background=True,
                                                    horizontal_foreground=True, background_grating=back_grate, 
                                                    circle_grating=circ_grate, n_steps=n_steps)
        post_vid = np.tile(back_grate[None], (post_frames,1,1))
        if i == 0:
            pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
            diff_vid = np.concatenate([pre_vid, differential, post_vid], axis=0)            
        else:
            diff_vid = np.concatenate([differential, post_vid], axis=0)
        vid.append(diff_vid)
        global_, back_grate, circ_grate = tdrstim.periodic_differential_circle(n_frames=diff_frames, 
                                                    period_dur=30, sync_periods=True, image_shape=img_shape,
                                                    center=center, radius=radius, horizontal_background=True,
                                                    horizontal_foreground=True, background_grating=back_grate, 
                                                    circle_grating=back_grate.copy(), n_steps=n_steps)
        post_vid = np.repeat(global_[-1:], post_frames, axis=0)
        global_vid = np.concatenate([global_, post_vid], axis=0)
        vid.append(global_vid)
    return np.concatenate(vid, axis=0)

def oms_differential(model, duration=5, sample_rate=30, pre_frames=40, post_frames=40, img_shape=(50,50), center=(25,25), radius=8, background_velocity=0, foreground_velocity=.5, seed=None, bar_size=2, inner_bar_size=None, filt_depth=40):
    """
    Plays a video of differential motion by keeping a circular window fixed in space on a 2d background grating.
    A grating exists behind the circular window that moves counter to the background grating. 

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_frames: int
        number of frames of still image to be prepended to the jittering
    post_frames: int
        number of frames of still image to be appended to the jittering
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window (0,0 is the upper left most pixel)
    radius: float
        the radius of the circular window
    background_velocity: float
        the magnitude of horizontal movement of the background grating in pixels per frame
    foreground_velocity: float
        the magnitude of horizontal movement of the foreground grating in pixels per frame
    seed: int or None
        sets the numpy random seed if int
    bar_size: int
        size of stripes. Min value is 3
    inner_bar_size: int
        size of grating bars inside circle. If None, set to bar_size
    """
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration * sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    differential, _, _ = tdrstim.differential_circle(diff_frames, bar_size=bar_size, inner_bar_size=inner_bar_size,
                                    foreground_velocity=foreground_velocity, 
                                    background_velocity=background_velocity,
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
    post_vid = np.repeat(differential[-1:], post_frames, axis=0)
    diff_vid = np.concatenate([pre_vid, differential, post_vid], axis=0)

    global_velocity = foreground_velocity if foreground_velocity != 0 else background_velocity
    global_, _, _ = tdrstim.differential_circle(diff_frames, bar_size=bar_size, inner_bar_size=inner_bar_size,
                                    foreground_velocity=global_velocity,
                                    background_velocity=global_velocity, 
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(global_[:1], pre_frames, axis=0)
    post_vid = np.repeat(global_[-1:], post_frames, axis=0)
    global_vid = np.concatenate([pre_vid, global_, post_vid], axis=0)
    
    if model is None:
        fig = None
        diff_response = None
        global_response = None
    else:
        x = torch.FloatTensor(tdrstim.concat(diff_vid, nh=filt_depth)).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        diff_response = resp.cpu().detach().numpy()

        x = torch.FloatTensor(tdrstim.concat(global_vid, nh=filt_depth)).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        global_response = resp.cpu().detach().numpy()

        # generate the figure
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(diff_response.mean(-1), color="g")
        ax.plot(global_response.mean(-1), color="b")
        ax.legend(["diff", "global"])
        diff_response = diff_response[pre_frames-40:tot_frames-post_frames]
        global_response = global_response[pre_frames-40:tot_frames-post_frames]
    return fig, diff_vid, global_vid, diff_response, global_response

def oms_jitter(model, duration=5, sample_rate=30, pre_frames=40, post_frames=40, img_shape=(50,50), center=(25,25), radius=5, seed=None, bar_size=2, inner_bar_size=None, jitter_freq=.5, step_size=1, filt_depth=40):
    """
    Plays a video of a jittered circle window onto a grating different than that of the background.

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_frames: int
        number of frames of still image to be prepended to the jittering
    post_frames: int
        number of frames of still image to be appended to the jittering
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window (0,0 is the upper left most pixel)
    radius: float
        the radius of the circular window
    seed: int or None
        sets the numpy random seed if int
    bar_size: int
        size of stripes. Min value is 3
    inner_bar_size: int
        size of stripes inside circle. Min value is 3. If none, same as bar_size
    jitter_freq: float between 0 and 1
        the frequency of jittered movements
    step_size: int
        largest magnitude jitter movements in pixels
    """
    assert pre_frames > 0 and post_frames > 0
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration * sample_rate)
    jitter_frames = int(tot_frames-pre_frames-post_frames)
    assert jitter_frames > 0
    jitters, _, _ = tdrstim.jittered_circle(jitter_frames, bar_size=bar_size, inner_bar_size=inner_bar_size, 
                                    foreground_jitter=jitter_freq, background_jitter=0, step_size=step_size,
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(jitters[:1], pre_frames, axis=0)
    post_vid = np.repeat(jitters[-1:], post_frames, axis=0)
    vid = np.concatenate([pre_vid, jitters, post_vid], axis=0)
    
    if model is None:
        fig = None
        response = None
    else:
        x = torch.FloatTensor(tdrstim.concat(vid, nh=filt_depth)).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        response = resp.cpu().detach().numpy()
        avg_response = response.mean(-1)

        # generate the figure
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(avg_response)
    return fig, vid, response

def oms(duration=5, sample_rate=0.01, transition_duration=0.07, silent_duration=0.93,
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
            object_mask = tdrstim.cmask(center, object_radius, object_frame)
            background_frame[object_mask] = object_frame[object_mask]

        # adjust contrast
        background_frame *= contrast
        movie[frame] = background_frame
    return movie


def osr(model=None, duration=2, interval=10, nflashes=5, intensity=-2.0, filt_depth=40):
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
    single_flash = tdrstim.flash(duration, interval, interval * 2, intensity=intensity)
    omitted_flash = tdrstim.flash(duration, interval, interval * 2, intensity=0.0)
    flash_group = list(repeat(single_flash, nflashes))
    zero_pad = np.zeros((interval, 1, 1))
    X = tdrstim.concat(zero_pad, *flash_group, omitted_flash, *flash_group, nx=50, nh=filt_depth)
    X[X!=0] = 1
    if model is not None:
        X_torch = torch.from_numpy(X).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(X_torch.shape[0]):
                    resp, hs = model(X_torch[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(X_torch)
        resp = resp.cpu().detach().numpy()
        figs = viz.response1D(X[:, -1, 0, 0].copy(), resp, figsize=(20, 8))
        (fig, (ax0,ax1)) = figs

        # Table Metrics
        n_full_responses = len(resp)//len(single_flash)
        responses = [resp[i*len(single_flash):(i+1)*len(single_flash)] for i in range(n_full_responses)]
        flash_resps = np.zeros((len(responses), *responses[0].shape))
        for i,resp in enumerate(responses):
            if i != len(flash_group):
                flash_resps[i] = resp
        omitted_resp = np.asarray(responses[len(flash_group)])
        avg_flash_resp = np.mean(flash_resps, axis=0)
        resp_ratio = (omitted_resp.sum(0)/avg_flash_resp.sum(0)).mean()
    else:
        fig = None
        ax0,ax1 = None, None
        resp = None
        resp_ratio = None

    return (fig, (ax0,ax1)), X, resp, resp_ratio

def motion_anticipation(model, scale_factor=55, velocity=0.08, width=2, flash_duration=2, filt_depth=40, make_fig=True):
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
    # c_right and c_left are the center positions of the bar
    c_right, speed_right, stim_right = tdrstim.driftingbar(velocity, width, x=(-30, 30))
    x = torch.from_numpy(stim_right).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(x)
    resp_right = resp.cpu().detach().numpy()

    c_left, speed_left, stim_left = tdrstim.driftingbar(-velocity, width, x=(30, -30))
    x = torch.from_numpy(stim_left).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(x)
    resp_left = resp.cpu().detach().numpy()

    # flashed bar stimulus
    flash_centers = np.arange(-25, 26)
    flashes = (tdrstim.flash(flash_duration, 43, 70, intensity=tdrstim.bar((x, 0), width, 50))
               for x in flash_centers)

    # flash responses are a 3-D array with dimensions (centers, stimulus time, cell)
    flash_responses = []
    with torch.no_grad():
        for f in tqdm(flashes):
            x = torch.from_numpy(tdrstim.concat(f, nh=filt_depth)).to(DEVICE)
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
            flash_responses.append(resp.cpu().detach().numpy())
        
    flash_responses = np.stack(flash_responses)

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

    if make_fig:
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

        return (fig, ax), (speed_left, speed_right), (c_right, stim_right, resp_right),(c_left, stim_left, resp_left), (flash_centers, flash_responses)#, (symmetry, continuity, peak_height, right_anticipation, left_anticipation)
    return (speed_left, speed_right), (c_right, stim_right, resp_right),(c_left, stim_left, resp_left), (flash_centers, flash_responses)#, (symmetry, continuity, peak_height, right_anticipation, left_anticipation)

def motion_reversal(model, scale_factor=55, velocity=0.08, width=2, filt_depth=40):
    """
    Moves a bar to the right and reverses it in the center, then does the same to the left. 
    The responses are averaged.
    Parameters
    ----------
    model : pytorch model
    scale_factor = 55       # microns per bar
    velocity = 0.08         # 0.08 bars/frame == 0.44mm/s, same as Berry et. al.
    width = 2               # 2 bars == 110 microns, Berry et. al. used 133 microns
    flash_duration = 2      # 2 frames == 20 ms, Berry et. al. used 15ms
    Returns
    -------
    motion : array_like
    flashes : array_like
    """
    # moving bar stimuli
    c_right, speed_right, stim_right = tdrstim.driftingbar(velocity, width)
    stim_right = stim_right[:,0]
    c_left, speed_left, stim_left = tdrstim.driftingbar(-velocity, width, x=(30, -30))
    stim_left = stim_left[:,0]
    # Find point that bars are at center
    right_halfway = None
    left_halfway = None 
    half_idx = stim_right.shape[1]//2
    for i in range(len(stim_right)):
        if right_halfway is None and stim_right[i,0, half_idx] <= -.99:
            right_halfway = i
        if left_halfway is None and stim_left[i, 0, half_idx] <= -.99:
            left_halfway = i
        if right_halfway is not None and left_halfway is not None:
            break
    # Create stimulus from moving bars
    rtl = np.concatenate([stim_right[:right_halfway], stim_left[left_halfway:]], axis=0)
    ltr = np.concatenate([stim_left[:left_halfway], stim_right[right_halfway:]], axis=0)
    if right_halfway < left_halfway:
        cutoff = left_halfway-right_halfway
        ltr = ltr[cutoff:-cutoff]
    elif left_halfway < right_halfway:
        cutoff = right_halfway-left_halfway
        rtl = rtl[cutoff:-cutoff]
 
    rtl_blocks = tdrstim.concat(rtl, nh=filt_depth)
    rtl_blocks = torch.from_numpy(rtl_blocks).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(rtl_blocks.shape[0]):
                resp, hs = model(rtl_blocks[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(rtl_blocks)
    resp_rtl = resp.cpu().detach().numpy()

    ltr_blocks = tdrstim.concat(ltr, nh=filt_depth)
    ltr_blocks = torch.from_numpy(ltr_blocks).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(ltr_blocks.shape[0]):
                resp, hs = model(ltr_blocks[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(ltr_blocks)
    resp_ltr = resp.cpu().detach().numpy()

    # average the response from multiple cells
    avg_resp_rtl = resp_rtl.mean(axis=-1)
    avg_resp_ltr = resp_ltr.mean(axis=-1)

    # normalize the average responses (to plot on the same scale)
    avg_resp_rtl /= avg_resp_rtl.max()
    avg_resp_ltr /= avg_resp_ltr.max()
    avg_resp = (avg_resp_rtl + avg_resp_ltr)/2

    # generate the figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    halfway = avg_resp_ltr.shape[0]//2
    ax.plot(np.arange(-halfway, halfway+1), avg_resp_ltr, 'g-', label='l->r')
    ax.plot(np.arange(-halfway, halfway+1), avg_resp_rtl, 'b-', label='r->l')
    ax.plot(np.arange(-halfway, halfway+1), avg_resp, 'r-', label='avg')
    ax.legend(frameon=True, fancybox=True, fontsize=18)
    ax.set_xlabel('Frames from reversal')
    ax.set_ylabel('Scaled firing rate')
    ax.set_xlim(-halfway, halfway)

    return (fig, ax), (speed_left, speed_right), (rtl, resp_rtl), (ltr, resp_ltr), avg_resp

def ds(img_shape=(500,500), bar_width=20, angle=0, step_size=10, n_repeats=3):
    frames = tdrstim.dir_select_vid(img_shape, bar_width=bar_width, angle=angle, step_size=step_size)
    ones = [1 for i in range(len(frames.shape[1:]))]
    return np.tile(frames, (n_repeats, *ones))

# Fast Contrast adaptation figure
# The following functions are used for the fast contrast adaptation figure
#######################################################################################

def repeat_white(time, contrast=1.0):
    compressed_time = int(np.ceil(time/3.0))
    compressed_stim = contrast * np.random.randn(compressed_time, 50, 50)
    stimulus = np.repeat(compressed_stim, 3, axis=0)
    return stimulus[:time]

def normalize_filter(sta, stimulus, target_sd, batch_size=1000):
    '''
    Enforces filtered stimulus to have the same standard deviation
    as the stimulus by scaling the values of the sta.
    '''
    with torch.no_grad():
        temp_sta = sta if type(sta) == type(torch.zeros(1)) else torch.FloatTensor(sta)
        temp_stim = stimulus if type(stimulus) == type(torch.zeros(1)) else\
                                                            torch.FloatTensor(stimulus)
        def sd_difference(theta):
            filt = abs(float(theta)) * temp_sta
            response = tdrutils.linear_response(filt, temp_stim, batch_size=batch_size,
                                                                         to_numpy=True)
            return (response.std() - target_sd)**2

        res = minimize(sd_difference, x0=1.0)
        theta = abs(res.x)
    return (theta * sta, theta, res.fun)

def filter_and_nonlinearity(model, contrast, layer_name='sequential.0',
                                                    unit_index=(0,15,15), 
                                                    nonlinearity_type='bin', 
                                                    filt_depth=40, sta=None, 
                                                    batch_size=2000,
                                                    verbose=False):
    # Computing STA
    if sta is None:
        if verbose:
            print("Calculating STA with contrast:", contrast)
        sta = tdrutils.compute_sta(model, contrast=contrast, layer=layer_name, 
                                                              cell_index=unit_index, 
                                                              n_samples=10000, 
                                                              batch_size=batch_size,
                                                              to_numpy=True, 
                                                              verbose=verbose)

    if verbose:
        print("Normalizing filter and collecting linear response")
    stimulus = repeat_white(9040, contrast)
    stimulus = tdrstim.rolling_window(stimulus, filt_depth)
    normed_sta, theta, error = normalize_filter(sta, stimulus, contrast, batch_size=batch_size)
    filtered_stim = tdrutils.linear_response(normed_sta, stimulus, batch_size=batch_size,
                                                                      to_numpy=True)

    # Inspecting model response
    if verbose:
        print("Collecting full model response")
    X = torch.FloatTensor(stimulus)
    model.eval()
    model_response = tdrutils.inspect(model, X, batch_size=batch_size, insp_keys={layer_name,},
                                                                 to_numpy=True,verbose=verbose)
    if type(unit_index) == type(int()):
        response = model_response[layer_name][:,unit_index]
    elif len(unit_index) == 1:
        response = model_response[layer_name][:,unit_index[0]]
    else:
        response = model_response[layer_name][:,unit_index[0], unit_index[1], unit_index[2]]

    # Fitting nonlinearity
    if verbose:
        print("Fitting Nonlinearity")
    if nonlinearity_type == 'bin':
        n_bins = 40
        nonlinearity = Binterp(n_bins)
    else:
        nonlinearity = Sigmoid()
    nonlinearity.fit(filtered_stim, response)

    time = np.linspace(0.4, 0, 40)
    #normed_sta = np.flip(normed_sta, axis=0)
    _, temporal = ft.decompose(normed_sta)
    temporal /= 0.01  # Divide by dt for y-axis to be s^{-1}

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), 40)
    nonlinear_prediction = nonlinearity.predict(x)

    return time, temporal, x, nonlinear_prediction

def contrast_fig(model, contrasts, layer_name=None, unit_index=0, verbose=False, 
                                                         nonlinearity_type='bin'):
    """
    Creates figure 3A from "Deeplearning Models Reveal..." paper. Much of this code has 
    been repurposed from Lane and Niru's notebooks. Significant chance of bugs...

    model: torch module
    contrasts: sequence of ints len 2 [low, high]
        the sequence should be in ascending order
    layer_name: string
        specifies the layer of interest, if None, the final layer is used
    unit_index: int or sequence of length 3
        specifies the unit of interest
    nonlinearity_type: string
        fits the nonlinearity to the specified type. allowed args are "bin" and "sigmoid".
    """
    if layer_name is None:
        layer_name = "sequential." + str(len(model.sequential)-1)
    if verbose:
        print("Making Fast Contr Fig for", layer_name, "unit:", unit_index)

    low_contr, high_contr = contrasts
    tup = filter_and_nonlinearity(model, low_contr, layer_name=layer_name,
                                      unit_index=unit_index, verbose=verbose,
                                      nonlinearity_type=nonlinearity_type)
    low_time, low_temporal, low_x, low_nl = tup

    tup = filter_and_nonlinearity(model, high_contr, layer_name=layer_name,
                                      unit_index=unit_index, verbose=verbose,
                                      nonlinearity_type=nonlinearity_type)
    high_time, high_temporal, high_x, high_nl = tup

    # Assure correct sign of decomp
    mean_diff = ((high_temporal-low_temporal)**2).mean()
    neg_mean_diff = ((high_temporal+low_temporal)**2).mean()
    if neg_mean_diff < mean_diff:
        high_temporal = -high_temporal

    # Plot the decomp
    fig = plt.figure(figsize=(8, 2))
    plt.subplot(1, 2, 1)
    plt.plot(low_time, low_temporal, label='Contrast = %02d%%' %(0.35 * contrasts[0] * 100), 
                                                                                    color='g')
    plt.plot(high_time, high_temporal, label='Contrast = %02d%%' %(0.35 * contrasts[1] * 100), 
                                                                                    color='b')
    plt.xlabel('Delay (s)', fontsize=14)
    plt.ylabel('Filter ($s^{-1}$)', fontsize=14)
    plt.text(0.2, -30, 'Low', color='g', fontsize=18)
    plt.text(0.2, -15, 'High', color='b', fontsize=18)
    
    # plt.legend()
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    plt.subplot(1, 2, 2)
    plt.plot(high_x, len(high_x) * [0], 'k--', alpha=0.4)
    plt.plot(high_x, high_nl, linewidth=3, color='b')
    plt.plot(low_x, low_nl, linewidth=3, color='g')
    plt.xlabel('Filtered Input', fontsize=14)
    plt.ylabel('Output (Hz)', fontsize=14)
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    majorLocator = MultipleLocator(1)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(0.5)
    
    ax1.xaxis.set_major_locator(majorLocator)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.xaxis.set_minor_locator(minorLocator)
    return fig

def nonlinearity_fig(model, contrast, layer_name=None, unit_index=0, verbose=False, 
                                                         nonlinearity_type='bin'):
    """
    Creates figure 2D from "Deeplearning Models Reveal..." paper. Much of this code has 
    been repurposed from Lane and Niru's notebooks.

    model: torch module
    contrast: int
        contrast to calculate nonlinearity
    layer_name: string
        specifies the layer of interest, if None, the final layer is used
    unit_index: int or sequence of length 3
        specifies the unit of interest
    nonlinearity_type: string
        fits the nonlinearity to the specified type. allowed args are "bin" and "sigmoid".
    """
    if layer_name is None:
        layer_name = "sequential." + str(len(model.sequential)-1)
    if verbose:
        print("Making Nonlinearity Fig for", layer_name, "unit:", unit_index)

    tup = filter_and_nonlinearity(model, contrast, layer_name=layer_name,
                                      unit_index=unit_index, verbose=verbose,
                                      nonlinearity_type=nonlinearity_type)
    resp_time, temporal_resp, resp_x, resp = tup

    fig = plt.figure(figsize=(8, 2))
    plt.plot(resp_x, len(resp_x) * [0], 'k--', alpha=0.4)
    plt.plot(resp_x, resp, linewidth=3, color='b')
    plt.xlabel('Filtered Input', fontsize=14)
    plt.ylabel('Output (Hz)', fontsize=14)
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    majorLocator = MultipleLocator(1)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(0.5)
    
    ax1.xaxis.set_major_locator(majorLocator)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.xaxis.set_minor_locator(minorLocator)
    return fig

#########################################################################################################

def retinal_phenomena_figs(model, verbose=True):
    figs = []
    fig_names = []
    metrics = dict()
    filt_depth = model.img_shape[0]

    (fig, (ax0,ax1)), X, resp = step_response(model, filt_depth=filt_depth)
    figs.append(fig)
    fig_names.append("step_response")
    metrics['step_response'] = None

    (fig,_),_,_,osr_resp_ratio = osr(model,duration=1,filt_depth=filt_depth)
    figs.append(fig)
    fig_names.append("osr")
    metrics['osr'] = osr_resp_ratio 

    (fig, (ax0,ax1)), X, resp = reversing_grating(model, filt_depth=filt_depth)
    figs.append(fig)
    fig_names.append("reversing_grating")
    metrics['reversing_grating'] = None

    (fig, (_)), _, _ = contrast_adaptation(model, .35, .05, filt_depth=filt_depth)
    figs.append(fig)
    fig_names.append("contrast_adaptation")
    metrics['contrast_adaptation'] = None

    contrasts = [0.05, 0.35]
    fig = contrast_fig(model, contrasts, unit_index=0, nonlinearity_type="bin", verbose=verbose)
    figs.append(fig)
    fig_names.append("fast_contr_adaptation")
    metrics['contrast_fig'] = None

    (fig, ax), _, _, _, _ = motion_reversal(model, filt_depth=filt_depth)
    figs.append(fig)
    fig_names.append("motion_reversal")
    metrics['motion_reversal'] = None

    tup = motion_anticipation(model, filt_depth=filt_depth)
    (fig, ax) = tup[0]
    figs.append(fig)
    fig_names.append("motion_anticipation")
    metrics['motion_anticipation'] = None

    tup = oms_random_differential(model, filt_depth=filt_depth)
    fig, _, _, diff_response, global_response = tup
    figs.append(fig)
    fig_names.append("oms")
    oms_ratios = global_response.mean(0)/diff_response.mean(0)
    metrics['oms'] = oms_ratios

    return figs, fig_names, metrics

