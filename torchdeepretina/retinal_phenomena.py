import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
from pyret.nonlinearities import Binterp, Sigmoid
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import collections
from tqdm import tqdm
from itertools import repeat
import torchdeepretina.batch_compute as bc
import torchdeepretina.stimuli as stim
import torchdeepretina.visualizations as viz
from tqdm import tqdm, trange
import torch

DEVICE = torch.device("cuda:0")

def step_response(model, duration=100, delay=50, nsamples=200, intensity=-1.):
    """Step response"""
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

def oms_random_differential(model, duration=4, sample_rate=0.01, pre_silent=.75, post_silent=.75, img_shape=(50,50), center=(25,25), radius=5, background_velocity=.4, foreground_velocity=.5, seed=None):
    """
    Plays a video of differential motion by keeping a circular window fixed in space on a 2d background grating.
    A grating exists behind the circular window that moves counter to the background grating. Each grating is jittered
    randomly.

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_silent: float
        duration of still image to be prepended to the jittering
    post_silent: float
        duration of still image to be appended to the jittering
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
    """
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration // sample_rate)
    pre_frames = int(pre_silent//sample_rate)
    post_frames = int(post_silent//sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    differential = stim.random_differential_circle(diff_frames, bar_size=4, 
                                    foreground_velocity=foreground_velocity, 
                                    background_velocity=background_velocity,
                                    image_shape=img_shape, center=center, radius=radius) 
    pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
    post_vid = np.repeat(differential[-1:], post_frames, axis=0)
    diff_vid = np.concatenate([pre_vid, differential, post_vid], axis=0)

    global_ = stim.random_differential_circle(diff_frames, bar_size=4, 
                                    foreground_velocity=foreground_velocity, sync_jitters=True,
                                    background_velocity=foreground_velocity, 
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(global_[:1], pre_frames, axis=0)
    post_vid = np.repeat(global_[-1:], post_frames, axis=0)
    global_vid = np.concatenate([pre_vid, global_, post_vid], axis=0)
    
    diff_response = model(torch.FloatTensor(stim.concat(diff_vid)).to(DEVICE)).cpu().detach().numpy()
    global_response = model(torch.FloatTensor(stim.concat(global_vid)).to(DEVICE)).cpu().detach().numpy()

    # generate the figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(diff_response.mean(-1), color="g")
    ax.plot(global_response.mean(-1), color="b")
    ax.legend(["diff", "global"])
    ax.axvline(x=pre_frames-40, color='r')
    ax.axvline(x=tot_frames-post_frames, color='r')
    diff_response = diff_response[pre_frames-40:tot_frames-post_frames]
    global_response = global_response[pre_frames-40:tot_frames-post_frames]
    return fig, diff_vid, global_vid, diff_response, global_response

def oms_differential(model, duration=4, sample_rate=0.01, pre_silent=.75, post_silent=.75, img_shape=(50,50), center=(25,25), radius=5, background_velocity=0, foreground_velocity=.5, seed=None):
    """
    Plays a video of differential motion by keeping a circular window fixed in space on a 2d background grating.
    A grating exists behind the circular window that moves counter to the background grating. 

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_silent: float
        duration of still image to be prepended to the jittering
    post_silent: float
        duration of still image to be appended to the jittering
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
    """
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration // sample_rate)
    pre_frames = int(pre_silent//sample_rate)
    post_frames = int(post_silent//sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    differential = stim.differential_circle(diff_frames, bar_size=4, 
                                    foreground_velocity=foreground_velocity, 
                                    background_velocity=background_velocity,
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
    post_vid = np.repeat(differential[-1:], post_frames, axis=0)
    diff_vid = np.concatenate([pre_vid, differential, post_vid], axis=0)

    global_ = stim.differential_circle(diff_frames, bar_size=4, 
                                    foreground_velocity=foreground_velocity,
                                    background_velocity=foreground_velocity, # Note the foreground velocity
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(global_[:1], pre_frames, axis=0)
    post_vid = np.repeat(global_[-1:], post_frames, axis=0)
    global_vid = np.concatenate([pre_vid, global_, post_vid], axis=0)
    
    diff_response = model(torch.FloatTensor(stim.concat(diff_vid)).to(DEVICE)).cpu().detach().numpy()
    global_response = model(torch.FloatTensor(stim.concat(global_vid)).to(DEVICE)).cpu().detach().numpy()

    # generate the figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(diff_response.mean(-1), color="g")
    ax.plot(global_response.mean(-1), color="b")
    ax.legend(["diff", "global"])
    ax.axvline(x=pre_frames-40, color='r')
    ax.axvline(x=tot_frames-post_frames, color='r')
    diff_response = diff_response[pre_frames-40:tot_frames-post_frames]
    global_response = global_response[pre_frames-40:tot_frames-post_frames]
    return fig, diff_vid, global_vid, diff_response, global_response

def oms_jitter(model, duration=4, sample_rate=0.01, pre_silent=.75, post_silent=.75, img_shape=(50,50), center=(25,25), radius=5, seed=None):
    """
    Plays a video of a jittered circle window onto a grating different than that of the background.

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_silent: float
        duration of still image to be prepended to the jittering
    post_silent: float
        duration of still image to be appended to the jittering
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window (0,0 is the upper left most pixel)
    radius: float
        the radius of the circular window
    seed: int or None
        sets the numpy random seed if int
    """
    assert pre_silent > 0 and post_silent > 0
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration // sample_rate)
    pre_frames = int(pre_silent//sample_rate)
    post_frames = int(post_silent//sample_rate)
    jitter_frames = int(tot_frames-pre_frames-post_frames)
    assert jitter_frames > 0
    jitters = stim.jittered_circle(jitter_frames, bar_size=4, foreground_jitter=.5, background_jitter=0,
                                    image_shape=img_shape, center=center, radius=radius, 
                                    horizontal_foreground=False, horizontal_background=False)
    pre_vid = np.repeat(jitters[:1], pre_frames, axis=0)
    post_vid = np.repeat(jitters[-1:], post_frames, axis=0)
    vid = np.concatenate([pre_vid, jitters, post_vid], axis=0)
    
    response = model(torch.FloatTensor(stim.concat(vid)).to(DEVICE)).cpu().detach().numpy()
    avg_response = response.mean(-1)

    # generate the figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(avg_response)
    ax.axvline(x=pre_frames-40, color='r')
    ax.axvline(x=tot_frames-post_frames, color='r')
    return fig, vid, response

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


def osr(model, duration=2, interval=10, nflashes=5, intensity=-2.0):
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

def motion_reversal(model, scale_factor=55, velocity=0.08, width=2):
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
    c_right, speed_right, stim_right = stim.driftingbar(velocity, width)
    stim_right = stim_right[:,0]
    c_left, speed_left, stim_left = stim.driftingbar(-velocity, width, x=(30, -30))
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
 
    rtl_blocks = stim.concat(rtl)
    rtl_blocks = torch.from_numpy(rtl_blocks).to(DEVICE)
    resp_rtl = model(rtl_blocks).cpu().detach().numpy()

    ltr_blocks = stim.concat(ltr)
    ltr_blocks = torch.from_numpy(ltr_blocks).to(DEVICE)
    resp_ltr = model(ltr_blocks).cpu().detach().numpy()

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

# Contrast adaptation figure
# The following functions are used for the contrast adaptation figure
#######################################################################################

def requires_grad(model, state):
    for p in model.parameters():
        try:
            p.requires_grad = state
        except:
            pass

def white(time, contrast=1.0):
    compressed_time = int(np.ceil(time/3.0))
    compressed_stim = contrast * np.random.randn(compressed_time, 50, 50)
    stimulus = np.repeat(compressed_stim, 3, axis=0)
    return stimulus[:time]

def get_stim_grad(model, X, layer, cell_idx, batch_size=500):
    """
    Gets the gradient of the model output at the specified layer and cell idx with respect
    to the inputs (X). Returns a gradient array with the same shape as X.
    """
    requires_grad(model, False)

    # Use hook to target appropriate layer activations
    outsize = (batch_size, 5)
    outs = torch.zeros(outsize).to(0)
    def forward_hook(module, inps, outputs):
        outs[:] = outputs
    hook_handles = []
    for name, module in model.named_modules():
        if name == layer:
            print("hook attached to " + name)
            hook_handles.append(module.register_forward_hook(forward_hook))

    # Get gradient with respect to activations
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i+batch_size].to(0)
        _ = model(x)
        # Outs are the activations at the argued layer and cell idx accross the batch
        if type(cell_idx) == type(int()):
            fx = outs[:,cell_idx].mean()
        elif len(cell_idx) == 1:
            fx = outs[:,cell_idx[0]].mean()
        else:
            fx = outs[:,cell_idx[0], cell_idx[1], cell_idx[2]].mean()
        fx.backward()
        outs = torch.zeros(outsize).to(0)
    del outs
    del _
    # Remove hooks to avoid memory leaks
    for handle in hook_handles:
        print("hook detached")
        handle.remove()

    requires_grad(model, True)
    return X.grad.data.cpu().detach().numpy()

def compute_sta(model, contrast, layer, cell_index):
    """helper function to compute the STA using the model gradient"""
    # generate some white noise
    X = stim.concat(white(1040, contrast=contrast)).copy()
    X = torch.FloatTensor(X)
    X.requires_grad = True

    # compute the gradient of the model with respect to the stimulus
    drdx = get_stim_grad(model, X, layer, cell_index)

    # average over the white noise samples
    sta = drdx.mean(axis=0)

    del X
    return sta

def normalize_filter(sta, stimulus, target_sd):
    '''Enforces filtered stimulus to have the same standard deviation
    as the stimulus.'''
    def sd_difference(theta):
        response = ft.linear_response(abs(theta) * sta, stimulus)
        return (np.std(response) - target_sd)**2

    res = minimize(sd_difference, x0=1.0)
    theta = abs(res.x)
    return (theta * sta, theta, res.fun)

def filter_and_nonlinearity(model, contrast, layer_name='sequential.0',
                                  unit_index=(0,15,15), nonlinearity_type='bin'):
    print("Computing STA")
    sta = compute_sta(model, contrast, layer_name, unit_index)
    sta = np.flip(sta, axis=0)

    print("Normalizing filter and collecting response")
    stimulus = white(4040, contrast=contrast)
    normed_sta, theta, error = normalize_filter(sta, stimulus, 0.35 * contrast)
    filtered_stim = ft.linear_response(normed_sta, stimulus)

    print("Inspecting model response")
    stim_tensor = torch.FloatTensor(stim.concat(stimulus))
    model_response = bc.batch_compute_model_response(stim_tensor, model, 500, insp_keys={layer_name})
    if type(unit_index) == type(int()):
        response = model_response[layer_name][:,unit_index]
    elif len(unit_index) == 1:
        response = model_response[layer_name][:,unit_index[0]]
    else:
        response = model_response[layer_name][:,unit_index[0], unit_index[1], unit_index[2]]

    print("Fitting nonlinearity")
    if nonlinearity_type == 'bin':
        nonlinearity = Binterp(80)
    else:
        nonlinearity = Sigmoid()
    nonlinearity.fit(filtered_stim[40:], response)

    print("Summarizing model for plotting")
    time = np.linspace(0.4, 0, 40)
    _, temporal = ft.decompose(normed_sta)
    temporal /= 0.01  # Divide by dt for y-axis to be s^{-1}

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), 40)
    nonlinear_prediction = nonlinearity.predict(x)

    return time, temporal, x, nonlinear_prediction

def contrast_fig(model, contrasts, layer_name=None, unit_index=(0,15,15), nonlinearity_type='bin'):
    """
    Creates figure 3A from "Deeplearning Models Reveal..." paper. Much of this code has been repurposed
    from Lane and Niru's notebooks. Significant chance of bugs...

    model: torch module
    contrasts: sequence of ints len 2
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
    low_time, low_temporal, low_x, low_nl = filter_and_nonlinearity(model, contrasts[0], layer_name=layer_name, 
                                                    unit_index=unit_index, nonlinearity_type=nonlinearity_type)
    high_time, high_temporal, high_x, high_nl = filter_and_nonlinearity(model, contrasts[1], layer_name=layer_name,
                                                    unit_index=unit_index, nonlinearity_type=nonlinearity_type)
    fig = plt.figure(figsize=(8, 2))
    plt.subplot(1, 2, 1)
    plt.plot(low_time, low_temporal, label='Contrast = %02d%%' %(0.35 * contrasts[0] * 100), color='g')
    plt.plot(high_time, high_temporal, label='Contrast = %02d%%' %(0.35 * contrasts[1] * 100), color='b')
    plt.xlabel('Delay (s)', fontsize=14)
    plt.ylabel('Filter ($s^{-1}$)', fontsize=14)
    plt.text(0.2, -15, 'High', color='b', fontsize=18)
    plt.text(0.2, -30, 'Low', color='g', fontsize=18)
    
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

#########################################################################################################
