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
import torch.nn as nn
import matplotlib.cm as cm
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.interpolate import interp1d
from tqdm import tqdm
try:
    from jetpack import errorplot
except:
    print("f2_response is unavailable until you run:\n$ pip install -e git+git://github.com/nirum/jetpack.git@master#egg=jetpack")
            
import deepdish as dd

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
device = DEVICE

def dirpref(rf, v, fr, n_steps=32):
    """
    This function computes the direction selectivity index and
    orientation selectivity index for a spatiotemporal receptive field
    It computes the dot product of an x-y-t receptive field with a set
    of sinusoidal gratings for one velocity and spatial frequency

    Stephen Baccus 12/2019
    Inputs:
        rf: ndarray or torch Tensor (T,H,W)
            the receptive field
        v: float
            velocity
        fr: float
            spatial frequency
        n_steps: int divisible by 4
            the number of discrete steps in the period of 0<=x<2π
    Outputs:
        dsi: direction selectivity index
        osi: orientation selectivity index
        resp_ang: response amplitude for each angle of motion
    """
    if isinstance(rf,np.ndarray):
        rf = torch.FloatTensor(rf)
    rf = rf.to(DEVICE)
    n_steps = int((n_steps//4)*4)
    # Using torch.meshgrid gives different results (which is absolutely
    # fucking stupid. Make me boil)
    ####rf = rf.permute((1,2,0))
    ####tup = np.meshgrid(np.arange(1,rf.shape[0]+1),
    ####                  np.arange(1,rf.shape[1]+1),
    ####                  np.arange(1,rf.shape[2]+1))
    ####X,Y,T = (torch.FloatTensor(arr).to(DEVICE) for arr in tup)
    tup = np.meshgrid(np.arange(1,rf.shape[1]+1),
                      np.arange(1,rf.shape[0]+1),
                      np.arange(1,rf.shape[2]+1))
    X,T,Y = (torch.FloatTensor(arr).to(DEVICE) for arr in tup)

    period = 2*np.pi
    pi_range = torch.linspace(0, period-period/n_steps, n_steps).to(DEVICE)
    resp_ang = []
    for i,ang in enumerate(pi_range):
        resp = []
        for j,phase in enumerate(pi_range):
            # gst: grating stimulus
            # fr : spatial frequency
            # ang: motion angle
            # phase: grating position
            # X, Y, T: location in space-time rf   
            gst = fr*torch.cos(ang)*X + fr*torch.sin(ang)*Y - T*v + phase
            gst = torch.sin(gst)
            prod = (gst*rf).sum()
            resp.append(prod.item())

        diff = np.max(resp)-np.min(resp)
        resp_ang.append(diff)

    # prefdir: Preferred direction (prefdir) of the rf,
    #          direction that has the largest response amplitude
    # nulldir: Null direction is opposite to preferred direction
    # dsi: Direction selectivity index
    half_steps = n_steps//2
    prefdir = np.argmax(resp_ang)
    nulldir = ((prefdir+half_steps) % n_steps)
    dsi = (resp_ang[prefdir]-resp_ang[nulldir]) / resp_ang[prefdir]

    # Compute orientation selectivity index
    # orthdir1 & 2:Orthogonal directions from preferred-null direction 
    # osi: Orientation selectivity index
    # respprefor: Mean response for preferred orientation
    # respnulllor: Mean response for orthogonal orientation
    quart_steps = n_steps//4
    orthdir1 = (prefdir+quart_steps)   % n_steps
    orthdir2 = (prefdir+3*quart_steps) % n_steps
    respprefor = (resp_ang[prefdir]+resp_ang[nulldir])/2
    respnullor = (resp_ang[orthdir1]+resp_ang[orthdir2])/2
    osi = (respprefor-respnullor)/respprefor
    return dsi, osi, np.asarray(resp_ang)

def dsiosi_idx(rf, velocities=torch.arange(0.1,.71,0.1),
                  frequencies=torch.arange(0.1,1.11,0.1),
                  n_steps=32):
    """
    Computes direction selectivity index (DSI) and orientation
    selectivity index (OSI) for a receptive field for a range of
    velocities and spatial frequencies. It loops over the function
    dirpref, which computes indices for one stimulus.

    Stephen Baccus 12/2019
    Inputs:
        rf: ndarray or torch tensor (T,H,W)
            receptive field
        velocities: listlike of floats
            range of velocities
        frequencies: listlike of floats
            range of spatial frequencies
        n_steps: int divisible by 4
            the number of discrete steps in the period of 0<=x<2π
    
    Outputs:
        dsimax:
            maximum direction selectivity index
        osimax:
            maximum orientation selectivity index
        angmax:
            maximum angle
        respmax: ndarray
            the response with the maximum response
    """
    n_steps = int((n_steps//4)*4)
    period = 2*np.pi
    # Compute DSI and OSI for the range of velocities and frequencies
    # DSI will vary with velocity and spatial frequency
    # if nV,nFR and nDir are the number of velocities and spatial
    # frequencies and directions, then
    # dsi: nV x nFR array of DSIs
    # osi: nV x nFR array of oSIs
    # resp_ang: nV x nFR x nDir array of response amplitudes
    shape = (len(velocities), len(frequencies), n_steps)
    dsis = np.zeros(shape[:2])
    osis = np.zeros(shape[:2])
    resp_angs = np.zeros(shape)

    for i,v in enumerate(velocities):
        for j,f in enumerate(frequencies):
            dsi,osi,resp_ang = dirpref(rf,v,f,n_steps=n_steps)

            dsis[i,j] = dsi
            osis[i,j] = osi
            resp_angs[i,j,:] = resp_ang

    # Find the stimulus velocity and spatial frequency with the maximum
    # response, and choose DSIs and OSIs for that stimulus
    maxresp = np.max(resp_angs,axis=2) # Max response across directions
                              # for each velocity,v and frequency, fr
    row,col,value = tdrutils.max_matrix(maxresp) # Max response across v and fr

    # DSI and OSI using grating stimulus of max response
    dsimax = dsis[row,col]
    osimax = osis[row,col]

    # Preferred direction using grating stimulus of max response
    respmax = resp_angs[row,col,:]
    angmax = np.argmax(respmax) * (period/n_steps)

    return dsimax,osimax,angmax,respmax,resp_angs

def step_response(model, duration=100, delay=50, nsamples=200,
                                intensity=-1., filt_depth=40):
    """
    Interrogates the model's step response

    model: torch Module
    duration: int
        the duration of the flashes
    delay: int
        the delay between flashes
    nsamples: int
        the length of the flash stimulus in frames
    intensity: float
        the numeric value of the flash pixels
    filt_depth: int
        the depth of the training stimulus in terms of frames
    """
    flash = tdrstim.flash(duration, delay, nsamples,
                                intensity=intensity)
    X = tdrstim.concat(flash, nh=filt_depth)
    X_torch = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs =[torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(X_torch.shape[0]):
                resp, hs = model(X_torch[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(X_torch)
    resp = resp.cpu().detach().numpy()
    figs = viz.response1D(X[:,-1,0,0].copy(), resp)
    (fig, (ax0,ax1)) = figs
    return (fig, (ax0,ax1)), X, resp


def paired_flash(model, ifis=(2, 20), duration=1, intensity=-2.0,
                                            total=100, delay=40):
    """
    Generates responses to a pair of neighboring flashes

    Parameters
    ----------
    ifi : int
        inter-flash interval, in samples (default: 5)
    duration : int
        the duration of each flash in frames (default: 1)
    intensity : float
        the flash intensity (default: -2.0)
    padding : int
        how much padding in frames to put on either side of the
        flash (default: 50)
    """
    s1, r1, s2, r2 = [], [], [], []
    stimuli = []
    responses = []

    for ifi in tqdm(np.arange(ifis[0], ifis[1], duration)):
        # single flashes
        x1 = tdrstim.paired_flashes(ifi, duration, (intensity, 0),
                                                     total, delay)
        s1.append(tdrstim.unroll(x1)[:, 0, 0])
        x1_torch = torch.from_numpy(x1).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                             model.h_shapes]
                resps = []
                for i in range(x1_torch.shape[0]):
                    resp, hs = model(x1_torch[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x1_torch)
        r1.append(tdrstim.prepad(resp.cpu().detach().numpy()))

        x2 = tdrstim.paired_flashes(ifi, duration, (0, intensity),
                                                     total, delay)
        x2_torch = torch.from_numpy(x2).to(DEVICE)
        s2.append(tdrstim.unroll(x2)[:, 0, 0])
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                             model.h_shapes]
                resps = []
                for i in range(x2_torch.shape[0]):
                    resp, hs = model(x2_torch[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x2_torch)
        r2.append(tdrstim.prepad(resp.cpu().detach().numpy()))

        # pair
        x = tdrstim.paired_flashes(ifi, duration, intensity, total,\
                                                              delay)
        x_torch = torch.from_numpy(x).to(DEVICE)
        stimuli.append(tdrstim.unroll(x)[:, 0, 0])
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                             model.h_shapes]
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
    """
    A reversing grating stimulus

    model: torch Module
    size: int
        size of grating bars in pixels
    phase: float
        The phase of the grating in the x dimension (as a
        fraction of the period). Must be between 0 and 1.
    filt_depth: int
        the number of temporal frames in the stimulus that is fed
        into the model.
    """
    grating = tdrstim.grating(barsize=(size, 0), phase=(phase, 0.0),
                                               intensity=(1.0, 1.0),
                                               us_factor=1, blur=0)
    X = tdrstim.reverse(grating, halfperiod=50, nsamples=300)
    X = tdrstim.rolling_window(X,filt_depth)
    X_torch = torch.FloatTensor(X).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in\
                                            model.h_shapes]
            resps = []
            for i in range(X_torch.shape[0]):
                resp, hs = model(X_torch[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(X_torch)
    figs = viz.response1D(X[:, -1, 0, 0].copy(),
                                   resp.cpu().detach().numpy())
    (fig, (ax0,ax1)) = figs
    return (fig, (ax0,ax1)), X, resp

def contrast_adaptation(model, c0=0.7, c1=.1, tot_dur=300,
                                    filt_depth=40, nx=50):
    """
    Step change in contrast

    model: torch Module
    c0: float
        the first contrast
    c1: float
        the second contrast
    tot_dur: int
        the length of the stimulus in frames
    filt_depth: int
        the depth of the first convolutional filter in the model
    """

    # the contrast envelope
    qrtr_dur = int((tot_dur-filt_depth)//4)
    remainder = (tot_dur-filt_depth)%4
    flicker_1 = tdrstim.repeat_white(filt_depth+qrtr_dur, nx=nx,
                                        contrast=c0, n_repeats=3)
    flicker_2 = tdrstim.repeat_white(qrtr_dur*2, nx=nx, contrast=c1,
                                                        n_repeats=3)
    flicker_3 = tdrstim.repeat_white(qrtr_dur+remainder, nx=nx,
                                      contrast=c0, n_repeats=3)
    envelope = np.concatenate([flicker_1, flicker_2, flicker_3],
                                                         axis=0)

    # generate a bunch of responses to random noise with the given
    # contrast envelope
    with torch.no_grad():
        rand = np.random.randn(*envelope.shape) * envelope
        x = torch.from_numpy(tdrstim.concat(rand, nh=filt_depth))
        x = x.to(DEVICE)
        if model.recurrent:
            hs =[torch.zeros(1,*h).to(device) for h in model.h_shapes]
            resps = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(x)
    response = resp.detach().cpu().numpy()
    figs = viz.response1D(envelope[40:, 0, 0], response)
    (fig, (ax0,ax1)) = figs

    return (fig, (ax0,ax1)), envelope, response

def oms_random_differential(model, duration=5, sample_rate=30,
                                               pre_frames=40,
                                               post_frames=40,
                                               img_shape=(50,50),
                                               center=(25,25),
                                               radius=8,
                                               background_velocity=.3,
                                               foreground_velocity=.5,
                                               seed=None, bar_width=4,
                                               inner_bar_width=None,
                                               filt_depth=40):
    """
    Plays a video of differential motion by keeping a circular window
    fixed in space on a 2d background grating. A grating exists behind
    the circular window that moves counter to the background grating.
    Each grating is jittered randomly.

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_frames: int
        number of frames of still image to be prepended to the
        jittering
    post_frames: int
        number of frames of still image to be appended to the
        jittering
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window
        (0,0 is the upper left most pixel)
    radius: float
        the radius of the circular window
    background_velocity: float
        the intensity of the horizontal jittering of the background
        grating
    foreground_velocity: float
        the intensity of the horizontal jittering of the foreground
        grating
    seed: int or None
        sets the numpy random seed if int
    bar_width: int
        size of stripes. Min value is 3
    inner_bar_width: int
        size of grating bars inside circle. If None, set to bar_width
    """
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration * sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    differential,_,_ = tdrstim.random_differential_circle(diff_frames,
                              bar_width=bar_width,
                              inner_bar_width=inner_bar_width,
                              foreground_velocity=foreground_velocity,
                              background_velocity=background_velocity,
                              image_shape=img_shape,
                              center=center, radius=radius) 
    pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
    post_vid = np.repeat(differential[-1:], post_frames, axis=0)
    diff_vid =np.concatenate([pre_vid,differential,post_vid], axis=0)

    global_velocity = foreground_velocity if foreground_velocity != 0\
                                              else background_velocity
    global_, _, _ = tdrstim.random_differential_circle(diff_frames,
                                 bar_width=bar_width,
                                 inner_bar_width=inner_bar_width,
                                 foreground_velocity=global_velocity,
                                 sync_jitters=True,
                                 background_velocity=global_velocity,
                                 image_shape=img_shape, center=center,
                                 radius=radius, 
                                 horizontal_foreground=False,
                                 horizontal_background=False)
    pre_vid = np.repeat(global_[:1], pre_frames, axis=0)
    post_vid = np.repeat(global_[-1:], post_frames, axis=0)
    global_vid = np.concatenate([pre_vid, global_, post_vid], axis=0)

    if model is None:
        fig = None
        diff_response = None
        global_response = None
    else:
        x = torch.FloatTensor(tdrstim.rolling_window(diff_vid,
                                                     filt_depth))
        x = x.to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                                model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        diff_response = resp.cpu().detach().numpy()

        x = tdrstim.rolling_window(global_vid, filt_depth)
        x = torch.FloatTensor(x).to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                                    model.h_shapes]
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
        s = np.s_[pre_frames-40:tot_frames-post_frames]
        diff_response = diff_response[s]
        global_response = global_response[s]
    return fig, diff_vid, global_vid, diff_response, global_response

def oms_differential(model, duration=5, sample_rate=30, pre_frames=40,
                                    post_frames=40, img_shape=(50,50),
                                    center=(25,25), radius=8,
                                    background_velocity=0,
                                    foreground_velocity=.5,
                                    seed=None, bar_width=2,
                                    inner_bar_width=None,
                                    filt_depth=40):
    """
    Plays a video of differential motion by keeping a circular
    window fixed in space on a 2d background grating. A grating exists
    behind the circular window that moves counter to the background
    grating. 

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_frames: int
        number of frames of still image to be prepended to the
        jittering
    post_frames: int
        number of frames of still image to be appended to the
        jittering
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window
        (0,0 is the upper left most pixel)
    radius: float
        the radius of the circular window
    background_velocity: float
        the magnitude of horizontal movement of the background grating
        in pixels per frame
    foreground_velocity: float
        the magnitude of horizontal movement of the foreground grating
        in pixels per frame
    seed: int or None
        sets the numpy random seed if int
    bar_width: int
        size of stripes. Min value is 3
    inner_bar_width: int
        size of grating bars inside circle. If None, set to bar_width
    """
    if seed is not None:
        np.random.seed(seed)
    tot_frames = int(duration * sample_rate)
    diff_frames = int(tot_frames-pre_frames-post_frames)
    assert diff_frames > 0
    tup = tdrstim.differential_circle(diff_frames, bar_width=bar_width,
                                    inner_bar_width=inner_bar_width,
                              foreground_velocity=foreground_velocity,
                              background_velocity=background_velocity,
                              image_shape=img_shape, center=center,
                              radius=radius,
                              horizontal_foreground=False,
                              horizontal_background=False)
    differential, _, _ = tup
    pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
    post_vid = np.repeat(differential[-1:], post_frames, axis=0)
    diff_vid = np.concatenate([pre_vid,differential,post_vid], axis=0)

    global_velocity = foreground_velocity if foreground_velocity != 0\
                                              else background_velocity
    global_, _, _ = tdrstim.differential_circle(diff_frames,
                                  bar_width=bar_width,
                                  inner_bar_width=inner_bar_width,
                                  foreground_velocity=global_velocity,
                                  background_velocity=global_velocity,
                                  image_shape=img_shape,
                                  center=center, radius=radius,
                                  horizontal_foreground=False,
                                  horizontal_background=False)
    pre_vid = np.repeat(global_[:1], pre_frames, axis=0)
    post_vid = np.repeat(global_[-1:], post_frames, axis=0)
    global_vid = np.concatenate([pre_vid, global_, post_vid], axis=0)
    
    if model is None:
        fig = None
        diff_response = None
        global_response = None
    else:
        x = torch.FloatTensor(tdrstim.rolling_window(diff_vid,
                                                     filt_depth))
        x = x.to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                                model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
        diff_response = resp.cpu().detach().numpy()

        x = torch.FloatTensor(tdrstim.rolling_window(global_vid,
                                                     filt_depth))
        x = x.to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                             model.h_shapes]
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
        s = np.s_[pre_frames-40:tot_frames-post_frames]
        diff_response = diff_response[s]
        global_response = global_response[s]
    return fig, diff_vid, global_vid, diff_response, global_response

def oms_jitter(model, duration=5, sample_rate=30, pre_frames=40,
                              post_frames=40, img_shape=(50,50),
                              center=(25,25), radius=5,
                              seed=None, bar_width=2,
                              inner_bar_width=None, jitter_freq=.5,
                              step_size=1, filt_depth=40):
    """
    Plays a video of a jittered circle window onto a grating different
    than that of the background.

    duration: float
        length of video in seconds
    sample_rate: float
        sample rate of video in frames per second
    pre_frames: int
        number of frames of still image to be prepended to the jitter
    post_frames: int
        number of frames of still image to be appended to the jitter
    img_shape: sequence of ints len 2
        the image size (H,W)
    center: sequence of ints len 2
        the starting pixel coordinates of the circular window (0,0 is
        the upper left most pixel)
    radius: float
        the radius of the circular window
    seed: int or None
        sets the numpy random seed if int
    bar_width: int
        size of stripes. Min value is 3
    inner_bar_width: int
        size of stripes inside circle. Min value is 3. If none, same
        as bar_width
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
    jitters, _, _ = tdrstim.jittered_circle(jitter_frames,
                                    bar_width=bar_width,
                                    inner_bar_width=inner_bar_width,
                                    foreground_jitter=jitter_freq,
                                    background_jitter=0,
                                    step_size=step_size,
                                    image_shape=img_shape,
                                    center=center, radius=radius,
                                    horizontal_foreground=False,
                                    horizontal_background=False)
    pre_vid = np.repeat(jitters[:1], pre_frames, axis=0)
    post_vid = np.repeat(jitters[-1:], post_frames, axis=0)
    vid = np.concatenate([pre_vid, jitters, post_vid], axis=0)
    
    if model is None:
        fig = None
        response = None
    else:
        x = torch.FloatTensor(tdrstim.rolling_window(vid, filt_depth))
        x = x.to(DEVICE)
        with torch.no_grad():
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                             model.h_shapes]
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

def oms(duration=5, sample_rate=0.01, transition_duration=0.07,
                                          silent_duration=0.93,
                                          magnitude=5, space=(50, 50),
                                          center=(25, 25),
                                          object_radius=5,
                                          coherent=False, roll=False):
    """
    Object motion sensitivity stimulus, where an object moves
    differentially from the background.

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
    obj_position = np.hstack([np.zeros((silent_frames,)),
                         np.linspace(0, magnitude, transition_frames),
                         magnitude*np.ones((silent_frames,)),
                         np.linspace(magnitude, 0,transition_frames)])
    obj_position = obj_position.astype('int')

    half_silent = silent_frames // 2
    back_position = np.hstack([obj_position[half_silent:],
                             obj_position[:-half_silent]])
    back_position = back_position.astype('int')

    # make position sequence last total_frames
    if len(back_position) > total_frames:
        s = "Warning: movie won't be {} shorter than a full period."
        f = np.float(2 * transition_frames + 2 * silent_frames)
        back_position[:total_frames]
        obj_position[:total_frames]
    else:
        reps = int(np.ceil(np.float(total_frames)/len(back_position)))
        back_position = np.tile(back_position, reps)[:total_frames]
        obj_position = np.tile(obj_position, reps)[:total_frames]

    # create a larger fixed world of bars that we crop from later
    padding = 2 * grating_width + magnitude
    fixed_world = -1 * np.ones((space[0], space[1] + padding))
    for i in range(grating_width):
        fixed_world[:, i::2 * grating_width] = 1

    # make movie
    movie = np.zeros((total_frames, space[0], space[1]))
    for frame in range(total_frames):
        # make background grating
        s =np.s_[:,back_position[frame]:back_position[frame]+space[0]]
        background_frame = np.copy(fixed_world[s])

        if not coherent:
            # make object frame
            temp = obj_position[frame]+space[0]
            s = np.s_[:,obj_position[frame]:temp]
            object_frame = np.copy(fixed_world[s])

            # set center of background frame to object
            object_mask = tdrstim.cmask(center, object_radius,
                                                    object_frame)
            background_frame[object_mask] = object_frame[object_mask]

        # adjust contrast
        background_frame *= contrast
        movie[frame] = background_frame
    return movie


def osr_stim(duration=2, interval=8, nflashes=3, intensity=-1.0,
                                                 filt_depth=40,
                                                 noise_std=0.1):
    """
    Omitted stimulus response

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

    single_flash = tdrstim.flash(duration, interval,
                                           duration+interval,
                                           intensity=intensity)
    single_flash += noise_std*np.random.randn(*single_flash.shape)
    omitted_flash = tdrstim.flash(duration, interval,
                                            duration+interval,
                                            intensity=0.0)
    omitted_flash += noise_std*np.random.randn(*omitted_flash.shape)
    flash_group = list(repeat(single_flash, nflashes))

    zero_pad = np.zeros((filt_depth-interval, 1, 1))
    start_pad = np.zeros((interval * (nflashes-1), 1, 1))
    X = tdrstim.concat(start_pad, zero_pad, *flash_group,
                                            omitted_flash,
                                            zero_pad,
                                            nx=50,
                                            nh=filt_depth)
    omitted_idx = len(start_pad)+len(zero_pad)
    omitted_idx += nflashes*len(single_flash) + interval - filt_depth
    return X, omitted_idx

def motion_anticipation(model, scale_factor=55, velocity=0.08,
                                    width=2, flash_duration=2,
                                    filt_depth=40, make_fig=True):
    """Generates the Berry motion anticipation stimulus
    Stimulus from the paper:
    Anticipation of moving stimuli by the retina,
    M. Berry, I. Brivanlou, T. Jordan and M. Meister, Nature 1999
    Parameters
    ----------
    model : keras.Model
    scale_factor = 55       # microns per bar
    velocity = 0.08         # 0.08 bars/frame == 0.44mm/s, same as
                            # Berry et. al.
    width = 2               # 2 bars == 110 microns, Berry et. al.
                            # used 133 microns
    flash_duration = 2      # 2 frames==20ms, Berry et. al. used 15ms
    Returns
    -------
    motion : array_like
    flashes : array_like
    """
    # moving bar stimulus and responses
    # c_right and c_left are the center positions of the bar
    c_right, speed_right, stim_right = tdrstim.driftingbar(velocity,
                                                 width, x=(-30, 30))
    x = torch.from_numpy(stim_right).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in\
                                         model.h_shapes]
            resps = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)
        else:
            resp = model(x)
    resp_right = resp.cpu().detach().numpy()

    c_left, speed_left, stim_left = tdrstim.driftingbar(-velocity,
                                               width, x=(30, -30))
    x = torch.from_numpy(stim_left).to(DEVICE)
    with torch.no_grad():
        if model.recurrent:
            hs = [torch.zeros(1,*h).to(device) for h in\
                                         model.h_shapes]
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
    flashes = (tdrstim.flash(flash_duration, 43, 70,
                        intensity=tdrstim.bar((x, 0), width, 50))\
                        for x in flash_centers)

    # flash responses are a 3-D array with dimensions
    # (centers, stimulus time, cell)
    flash_responses = []
    with torch.no_grad():
        for f in tqdm(flashes):
            x = torch.from_numpy(tdrstim.concat(f, nh=filt_depth))
            x = x.to(DEVICE)
            if model.recurrent:
                hs = [torch.zeros(1,*h).to(device) for h in\
                                             model.h_shapes]
                resps = []
                for i in range(x.shape[0]):
                    resp, hs = model(x[i:i+1], hs)
                    resps.append(resp)
                resp = torch.cat(resps, dim=0)
            else:
                resp = model(x)
            flash_responses.append(resp.cpu().detach().numpy())
        
    flash_responses = np.stack(flash_responses)

    # pick off the flash responses at a particular time point
    # (the time of the max response)
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
        ax.plot(scale_factor * c_left[40:], avg_resp_left, 'g-',
                                            label='Left motion')
        ax.plot(scale_factor * c_right[40:], avg_resp_right, 'b-',
                                             label='Right motion')
        ax.plot(scale_factor * flash_centers, avg_resp_flash, 'r-',
                                                     label='Flash')
        ax.legend(frameon=True, fancybox=True, fontsize=18)
        ax.set_xlabel('Position ($\mu m$)')
        ax.set_ylabel('Scaled firing rate')
        ax.set_xlim(-735, 135)

        return (fig, ax), (speed_left, speed_right),\
                (c_right, stim_right, resp_right),\
                (c_left, stim_left, resp_left),\
                (flash_centers, flash_responses)
    return (speed_left, speed_right),\
            (c_right, stim_right, resp_right),\
            (c_left, stim_left, resp_left),\
            (flash_centers, flash_responses)

def motion_reversal(model, scale_factor=55, velocity=0.08, width=2,
                                                    filt_depth=40):
    """
    Moves a bar to the right and reverses it in the center, then does
    the same to the left.  The responses are averaged.

    Parameters
    ----------
    model : pytorch model
    scale_factor = 55       # microns per bar
    velocity = 0.08         # 0.08 bars/frame == 0.44mm/s, same as
                            # Berry et. al.
    width = 2               # 2 bars == 110 microns, Berry et. al.
                            # used 133 microns
    flash_duration = 2      # 2 frames==20ms, Berry et. al. used 15ms
    Returns
    -------
    motion : array_like
    flashes : array_like
    """
    # moving bar stimuli
    c_right, speed_right, stim_right = tdrstim.driftingbar(velocity,
                                                              width)
    stim_right = stim_right[:,0]
    c_left, speed_left, stim_left = tdrstim.driftingbar(-velocity,
                                               width, x=(30, -30))
    stim_left = stim_left[:,0]
    # Find point that bars are at center
    right_halfway = None
    left_halfway = None 
    half_idx = stim_right.shape[1]//2
    for i in range(len(stim_right)):
        if right_halfway is None and stim_right[i,0, half_idx]<=-.99:
            right_halfway = i
        if left_halfway is None and stim_left[i, 0, half_idx] <= -.99:
            left_halfway = i
        if right_halfway is not None and left_halfway is not None:
            break
    # Create stimulus from moving bars
    arr = [stim_right[:right_halfway], stim_left[left_halfway:]]
    rtl = np.concatenate(arr, axis=0)
    arr = [stim_left[:left_halfway], stim_right[right_halfway:]]
    ltr = np.concatenate(arr, axis=0)
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
            hs =[torch.zeros(1,*h).to(device) for h in model.h_shapes]
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
            hs =[torch.zeros(1,*h).to(device) for h in model.h_shapes]
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
    ax.plot(np.arange(-halfway, halfway+1), avg_resp_ltr, 'g-',
                                                  label='l->r')
    ax.plot(np.arange(-halfway, halfway+1), avg_resp_rtl, 'b-',
                                                  label='r->l')
    ax.plot(np.arange(-halfway, halfway+1), avg_resp, 'r-',
                                               label='avg')
    ax.legend(frameon=True, fancybox=True, fontsize=18)
    ax.set_xlabel('Frames from reversal')
    ax.set_ylabel('Scaled firing rate')
    ax.set_xlim(-halfway, halfway)

    return (fig, ax), (speed_left, speed_right), (rtl, resp_rtl),\
                                        (ltr, resp_ltr), avg_resp


# Fast Contrast adaptation figure
# The following functions are used for the fast contrast adaptation
# figure
######################################################################

def normalize_filter(sta, stimulus, target_sd, batch_size=1000):
    '''
    Enforces filtered stimulus to have the same standard deviation
    as the stimulus by scaling the values of the sta.
    '''
    with torch.no_grad():
        temp_sta = sta if type(sta) == type(torch.zeros(1)) else\
                                           torch.FloatTensor(sta)
        temp_stim = stimulus if type(stimulus)==type(torch.zeros(1))\
                                      else torch.FloatTensor(stimulus)
        def sd_difference(theta):
            filt = abs(float(theta)) * temp_sta
            response = tdrutils.linear_response(filt, temp_stim,
                                          batch_size=batch_size,
                                          to_numpy=True)
            return (response.std() - target_sd)**2

        res = minimize(sd_difference, x0=1.0)
        theta = abs(res.x)
    return (theta * sta, theta, res.fun)

def filter_and_nonlinearity(model, contrast,layer_name='sequential.0',
                                              unit_index=(0,15,15),
                                              nonlinearity_type='bin',
                                              filt_depth=40, sta=None,
                                              batch_size=2000,
                                              n_samples=10000,
                                              verbose=False):
    """
    Creates a filter and a nonlinearity fit to the model output at the
    specified layer. Used to evaluate the internal cells of the model.

    contrast: float
        intensity (std) of stimulus for fitting filter and
        nonlinearity
    layer_name: str
        the name of the layer of interest
    unit_index: tuple of ints (channel, row, col)
        the index of the unit of interest within the layer.
    nonlinearity_type: str
        the type of nonlinearity to be fit
    filt_depth: int
        the depth of the stimulus for each sample
    sta: ndarray (C,H,W)
        a premade spike triggered average
    batch_size: int
        the size of the batches used for computing the filter
    n_samples: int
        the number of samples to use when fitting the filter
    """
    # Computing STA
    stimulus = tdrstim.repeat_white(n_samples,nx=model.img_shape[1],
                                                  contrast=contrast,
                                                  n_repeats=3)
    stimulus = tdrstim.rolling_window(stimulus, model.img_shape[0])
    if sta is None:
        if verbose:
            print("Calculating STA with contrast:", contrast)
        sta = tdrutils.compute_sta(model, contrast=contrast,
                                           layer=layer_name,
                                           cell_index=unit_index,
                                           n_samples=n_samples,
                                           X=stimulus,
                                           batch_size=batch_size,
                                           to_numpy=True,
                                           verbose=verbose)
    if verbose:
        print("Normalizing filter and collecting linear response")
    normed_sta, theta, error = normalize_filter(sta, stimulus,
                                                contrast,
                                                batch_size=batch_size)
    filtered_stim = tdrutils.linear_response(normed_sta, stimulus,
                                            batch_size=batch_size,
                                            to_numpy=True)

    # Inspecting model response
    if verbose:
        print("Collecting full model response")
    X = torch.FloatTensor(stimulus)
    model.eval()
    model_response = tdrutils.inspect(model, X, batch_size=batch_size,
                                              insp_keys={layer_name,},
                                              to_numpy=True,
                                              verbose=verbose)
    if type(unit_index) == type(int()):
        response = model_response[layer_name][:,unit_index]
    elif len(unit_index) == 1:
        response = model_response[layer_name][:,unit_index[0]]
    else:
        response = model_response[layer_name][:,unit_index[0],
                                                unit_index[1],
                                                unit_index[2]]

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

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), )
    nonlinear_prediction = nonlinearity.predict(x)

    return time, temporal, x, nonlinear_prediction, nonlinearity

def contrast_fig(model, contrasts=[0.4,2.4], layer_name=None,
                                 unit_index=0, verbose=False,
                                 nonlinearity_type='bin'):
    """
    Creates figure 3A from "Deeplearning Models Reveal..." paper.
    Much of this code has been repurposed from Lane and Niru's
    notebooks. Significant chance of bugs...

    model: torch module
    contrasts: sequence of ints len 2 [low, high]
        the sequence should be in ascending order
    layer_name: string
        specifies the layer of interest, if None, the final layer is
        used
    unit_index: int or sequence of length 3
        specifies the unit of interest
    nonlinearity_type: string
        fits the nonlinearity to the specified type. allowed args are
        "bin" and "sigmoid".
    """
    if layer_name is None:
        layer_name = "sequential." + str(len(model.sequential)-1)
    if verbose:
        print("Making Fast Contr Fig for", layer_name, "unit:",
                                                    unit_index)

    low_contr, high_contr = contrasts
    tup = filter_and_nonlinearity(model, low_contr,
                                  layer_name=layer_name,
                                  unit_index=unit_index,
                                  verbose=verbose,
                                  nonlinearity_type=nonlinearity_type)
    low_time, low_temporal, low_x, low_nl, low_nonlinearity = tup

    tup = filter_and_nonlinearity(model, high_contr,
                                  layer_name=layer_name,
                                  unit_index=unit_index,
                                  verbose=verbose,
                                  nonlinearity_type=nonlinearity_type)
    high_time, high_temporal, high_x, high_nl, high_nonlinearity = tup

    # Assure correct sign of decomp
    mean_diff = ((high_temporal-low_temporal)**2).mean()
    neg_mean_diff = ((high_temporal+low_temporal)**2).mean()
    if neg_mean_diff < mean_diff:
        high_temporal = -high_temporal

    # Plot the decomp
    fig = plt.figure(figsize=(8, 2))
    plt.subplot(1, 2, 1)
    time = low_time[5:]
    temporal = low_temporal[5:]
    label = 'Contrast = %02d%%'%(contrasts[0]*100)
    plt.plot(time,temporal,label=label, color='g', linewidth=3)
    time = high_time[5:]
    temporal = high_temporal[5:]
    label = 'Contrast = %02d%%'%(contrasts[1]*100)
    plt.plot(time, temporal, label=label, color='b', linewidth=3)
    plt.xlabel('Delay (s)', fontsize=14)
    plt.ylabel('Filter ($s^{-1}$)', fontsize=14)
    plt.text(0.2, -30, 'Low', color='g', fontsize=18)
    plt.text(0.2, -15, 'High', color='b', fontsize=18)
    plt.xticks(ticks=[0.0,0.3])
    
    # plt.legend()
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    # Plot the nonlinearity
    plt.subplot(1, 2, 2)
    plt.locator_params(axis='x', nbins=3)
    plt.plot(high_x, len(high_x) * [0], 'k--', alpha=0.4)
    plt.plot(high_x, high_nl, linewidth=3, color='b')
    plt.plot(low_x, low_nl, linewidth=3, color='g')
    plt.xlabel('Filtered Input', fontsize=14)
    plt.ylabel('Output (Hz)', fontsize=14)
    plt.xticks(ticks=[-5, 5])
    plt.xlim([-10, 10])
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    return fig

def nonlinearity_fig(model, contrast=0.25, layer_name=None,
                                unit_index=0, verbose=False,
                                nonlinearity_type='bin',
                                figsize=(8,2)):
    """
    Creates figure 2D from "Deeplearning Models Reveal..." paper.
    Much of this code has been repurposed from Lane and Niru's
    notebooks.

    model: torch module
    contrast: int
        contrast to calculate nonlinearity
    layer_name: string
        specifies the layer of interest, if None, the final layer is
        used
    unit_index: int or sequence of length 3
        specifies the unit of interest
    nonlinearity_type: string
        fits the nonlinearity to the specified type. allowed args are
        "bin" and "sigmoid".
    """
    if layer_name is None:
        layer_name = "sequential." + str(len(model.sequential)-1)
    if verbose:
        print("Making Nonlinearity Fig for", layer_name, "unit:",
                                                      unit_index)

    tup = filter_and_nonlinearity(model, contrast,
                                 layer_name=layer_name,
                                 unit_index=unit_index,
                                 verbose=verbose,
                                 nonlinearity_type=nonlinearity_type)
    resp_time, temporal_resp, resp_x, resp, nonlinearity = tup

    fig = plt.figure(figsize=figsize)
    plt.plot(resp_x, len(resp_x) * [0], 'k--', alpha=0.4)
    plt.plot(resp_x, resp, linewidth=3, color='k')
    plt.xlabel('Filtered Input', fontsize=14)
    plt.ylabel('Output (Hz)', fontsize=14)
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    return fig

######################################################################
######################################################################
### OFFICIAL FIGURES FROM THE PAPER
######################################################################

def sqz_predict(mdls):
    """
    Squeeze predictions from multiple models into one array.

    mdls: list of torch Modules
    """
    def predict(X):
        X = torch.FloatTensor(X).cuda()
        return np.hstack([mdl(X).detach().cpu().numpy() for mdl in mdls])
    return predict


def freq_doubling(predict, ncells, phases, widths, halfperiod=25,
                                                    nsamples=140,
                                                    filt_depth=40):
    """
    
    """
    F1 = np.zeros((ncells, phases.size, widths.size))
    F2 = np.zeros_like(F1)

    period = (2 * halfperiod) * 0.01
    base_freq = 1 / period
    freqs = fftfreq(nsamples - filt_depth, 0.01)
    i1 = np.where(freqs == base_freq)[0][0]
    i2 = np.where(freqs == base_freq * 2)[0][0]

    for p, phase in enumerate(phases):
        for w, width in tqdm(list(enumerate(widths))):
            X = tdrstim.reversing_grating(nsamples, halfperiod, width,
                                                    phase, 'sin')
            stim = tdrstim.concat(X)
            r = predict(stim)
            for ci in range(r.shape[1]):
                amp = np.abs(fft(r[:, ci]))
                F1[ci, p, w] = amp[i1]
                F2[ci, p, w] = amp[i2]
    return F1, F2


def f2_response(models):
    """
    Generate f2 response figure.

    models: list of torch Modules or single torch Module
        either a list of models or a single model
    """

    if not isinstance(models, list):
        models = [models]
    ns_predict = sqz_predict(models)

    units = [m.n_units for m in models]
    ncells = np.sum(units)              # Total number of cells.
    phases = np.linspace(0, 1, 9)       # Grating phases.
    widths = np.arange(1, 9)            # Bar widths (checkers).
    sz = widths * 55.55                 # Bar widths (microns).
    F1, F2 = freq_doubling(ns_predict, ncells, phases, widths)
    fratio = F2 / F1
    mu = fratio.mean(axis=1)

    # Plot
    fig = plt.figure(figsize=(6,6))
    plt.style.use('deepretina')
    ax = fig.add_subplot(111)
    errorplot(sz, mu.mean(axis=0), mu.std(axis=0) / np.sqrt(ncells),
                                             method='line', fmt='o',
                                             color='#222222', ax=ax)
    ax.plot([0, 500], [1, 1], '--', color='lightgray', lw=1, zorder=0)
    #ax.set_xlabel('Bar width ($\mu m$)',fontsize=40)
    #ax.set_ylabel('$F_2/F_1$ component ratio',fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(ticks=[0,200,400])
    plt.yticks(ticks=[0,1,5])

    #ax.set_ylim(0, 8.2)
    plt.tight_layout()
    return fig

def signfix(taus):
    """Corrects sign of a temporal kernel."""
    new_taus = []
    for tau in taus:
        if np.mean(tau[25:]) > 0:
            new_taus.append(-tau)
        else:
            new_taus.append(tau)
    return np.stack(new_taus)


def cadapt(contrasts, mdls, ncells):
    """Runs contrast adaptation analysis."""
    
    celldata = []
    for mdl, nc in zip(mdls, ncells):
        mdl.cuda()
        mdl.eval()
        for ci in range(nc):
            taus = []
            for c in tqdm(contrasts):
                X = tdrstim.concat(tdrstim.white(200, nx=50, contrast=c))
                X = torch.FloatTensor(X).cuda()
                X.requires_grad = True
                pred = mdl(X)[:,ci]
                pred.sum().backward()
                g = X.grad.data.detach().cpu().numpy()
                rf = g.mean(axis=0)
                _, tau = ft.decompose(rf)
                taus.append(tau)
            celldata.append(signfix(np.stack(taus)))
        mdl.cpu()
    return np.stack(celldata)


def center_of_mass(f, p):
    return np.sum(f * p)

def upsample_single(tau, t, t_us):
    return interp1d(t, tau, kind='cubic')(t_us)


def upsample_tau(taus, t, t_us):
    return np.stack([upsample_single(tau, t, t_us) for tau in taus])


def npower(x):
    """Normalized Fourier power spectrum."""
    amp = np.abs(fft(x))
    power = amp ** 2
    return power / power.max()

def generate_data(model, min_contrast=0.5, max_contrast=2.0,
                                            n_contrasts=4):
    """Generates data for the contrast adaptation figure."""
    n_us = 1000
    contrasts = np.linspace(min_contrast, max_contrast, n_contrasts)
    t = np.linspace(-400, 0, model.img_shape[0])
    t_us = np.linspace(-400, 0, n_us)

    # Get temporal kernels
    ns = cadapt(contrasts,(model,),(model.n_units,))
    # Upsample kernels
    ns = ns.reshape(n_contrasts*model.n_units, -1)
    nss = upsample_tau(ns, t, t_us).reshape(model.n_units, n_contrasts,
                                                           n_us)
    Fns = np.stack([npower(ni) for ni in nss]) # FFT analysis

    freqs = fftfreq(n_us, 1e-3 * np.mean(np.diff(t_us)))

    data = {
        'ns': ns,
        'contrasts': contrasts,
        'nss': nss,
        't': t,
        't_us': t_us,
        'Fns': Fns,
        'freqs': freqs,
    }
    return data

def latenc(mdls, intensities):
    rates = []
    for i in tqdm(intensities):
        X = tdrstim.concat(tdrstim.flash(2, 40, 70, intensity=i))
        X = torch.FloatTensor(X).cuda()
        preds = []
        for mdl in mdls:
            back_to_cpu = False
            if not next(mdl.parameters()).is_cuda:
                back_to_cpu = True
                mdl.to(DEVICE)
            mdl.eval()
            #pred = mdl(X).detach().cpu().numpy()[:,i:i+1]
            pred = mdl(X).detach().cpu().numpy()
            preds.append(pred)
            if back_to_cpu:
                mdl.cpu()
        r = np.hstack(preds)
        rates.append(r)
    return np.stack(rates)

def upsample(r):
    t = np.linspace(0, 300, 30)
    ts = np.linspace(0, 200, 1000)
    return interp1d(t, r, kind='cubic')(ts)

def latency_encoding(models):
    if isinstance(models, nn.Module):
        models = [models]
    # Time and Contrasts.
    ts = np.linspace(0, 200, 1000)
    intensities = np.linspace(0, -10, 21)

    # Natural scenes.
    ns = latenc(models, intensities)
    ns_resp = ns.mean(axis=-1)
    ns_resps = np.stack([upsample(ri) for ri in ns_resp])
    ns_lats = ts[ns_resps.argmax(axis=1)]

    # Make figure.
    plt.style.use('deepretina')
    fig = plt.figure(figsize=(19, 6))

    # Panel 2: Natural scene responses.
    ax2 = fig.add_subplot(121)
    colors = cm.Reds(np.linspace(0, 1, len(ns)))
    for r, c in zip(ns_resps, colors):
        ax2.plot(ts, r, '-', color=c)
    ax2.set_xlabel('Time (s)',fontsize=40)
    ax2.set_ylabel('Response (Hz)',fontsize=40)
    ax2.set_title('Natural Scenes',fontsize=40)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.locator_params(nbins=4)
    ax2.tick_params(axis='both', which='major', labelsize=35)

    # Panel 3: Latency vs. Intensity
    ax3 = fig.add_subplot(122)
    nat_color = 'lightcoral'
    whit_color = '#888888'
    ax3.plot(-intensities, ns_lats, 'o', color=nat_color)
    ax3.set_xlabel('Intensity',fontsize=40)
    ax3.set_ylabel('Latency',fontsize=40)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.locator_params(nbins=4)
    ax3.tick_params(axis='both', which='major', labelsize=35)

    # Add labels.
    plt.xlabel('Intensity (a.u.)',fontsize=20)
    plt.ylabel('Latency (ms)',fontsize=20)
    plt.ylim(70, 100)
    plt.xlim(0, 10.5)

    plt.tight_layout()
    return fig

def fast_contrast_adaptation(model):
    plt.style.use('deepretina')
    data = generate_data(model)

    Fn = data['Fns']
    fr = fftfreq(1000, 4e-4)

    ns_cm = np.zeros((5, 4))
    for ci in range(5):
        for j in range(4):
            ns_cm[ci, j] = center_of_mass(fr[:7], Fn[ci, j, :7])

    c = data['contrasts']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    errorplot(c, ns_cm.mean(axis=0), ns_cm.std(axis=0) / np.sqrt(5), method='line', fmt='o', color='lightcoral', ax=ax)

    zs = np.linspace(0.5, 2.0, 1e3)
    Pn = np.polyfit(c, ns_cm.mean(axis=0), 1)

    ax.plot(zs, np.polyval(Pn, zs), '--', color='lightcoral')

    ax.set_xlim(0.25, 2.25)
    ax.set_ylim(9, 16)
    plt.xticks([])
    plt.yticks(ticks=[9,12,15])
    ax.set_xlabel('Contrast (a.u.)', fontsize=40)
    ax.set_ylabel('Frequency (Hz)', fontsize=40)
    ax.set_title('Center of mass of frequency response',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=35)
    #plt.locator_params(nbins=3)
    plt.tight_layout()
    return fig

def motion_sweep(xstop, speed=0.24):
    """
    Sweeps a bar across the screen.
    """
    c, _, X = tdrstim.driftingbar(speed=speed, x=(-40, xstop))
    centers = c[40:]
    return centers, tdrstim.concat(X)

def sqz(mdls, X):
    """
    Squeeze predictions from multiple models into one array.
    """
    X = torch.FloatTensor(X).cuda()
    preds = []
    for mdl in mdls:
        back_to_cpu = False
        if not next(mdl.parameters()).is_cuda:
            back_to_cpu = True
            mdl.to(DEVICE)
        mdl.eval()
        pred = mdl(X).detach().cpu().numpy()
        preds.append(pred)
        if back_to_cpu:
            mdl.cpu()
    r = np.hstack(preds)
    return r

def run_motion_reversal(x_locations, models, speed=0.19, clip_n=210,
                                                         scaling=1,
                                                         fps=0.01):
    """
    Gets responses to a bar reversing motion.
    """
    tflips, Xs = zip(*[tdrstim.dr_motion_reversal(xi,speed)[1:] for\
                                             xi in x_locations])
    data = [scaling*sqz(models, X) for X in tqdm(Xs)]
    time = np.linspace(-tflips[0], (len(Xs[0])-tflips[0]), len(Xs[0]))
    time = time*fps
    deltas = np.array(tflips) - np.array(tflips)[0]
    datacut = [data[0]]
    datacut += [data[i][deltas[i]:-deltas[i]] for i in range(1, len(deltas))]
    cn = int((datacut[0].shape[0]-clip_n)//2)
    t = time[cn:-cn]
    data_clipped = [d[cn:-cn] for d in datacut]
    mu = np.hstack(data_clipped).mean(axis=1)
    sig = np.hstack(data_clipped).std(axis=1) / np.sqrt(60)

    clip_n = len(data_clipped[0])
    #offset = 40//2
    #speed = 0.01
    #t = np.linspace(-clip_n//2 * speed , clip_n//2 * speed, clip_n) + offset*speed
    return t, mu, sig, deltas

def dr_motion_reversal(model):
    # Load natural scene models.
    speed = 0.19
    # Plot
    clip_n = 210
    fig = plt.figure(figsize=(10, 7))

    # Plot motion reversal response.
    xs = np.arange(-9, 3)
    t, mu, sig, deltas = run_motion_reversal(xs, (model,), speed, clip_n=clip_n, scaling=1)
    #sig = np.zeros_like(sig)
    ax = fig.add_subplot(111)
    nat_color = 'lightcoral'
    ax.plot(t,mu,color=nat_color, linewidth=4)
    errorplot(t, mu, sig, color='lightcoral', ax=ax, linewidth=5)

    ax.set_xlim(-.7, 1.1)
    ax.set_ylim(0, 14)
    ax.set_ylabel('Firing Rate (Hz)',fontsize=20)
    ax.set_xlabel('Time (s)',fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=35)
    plt.locator_params(nbins=3)
    plt.yticks(ticks=[0,5,10])
    plt.xticks(ticks=[-0.5, 0, 0.5, 1], labels=[-0.5, 0, 0.5, 1])
    l = plt.legend(['Natural Scenes'], fontsize=35, frameon=False,
                                                handlelength=0,
                                                markerscale=0)
    colors = [nat_color]
    for color,text in zip(colors, l.get_texts()):
        text.set_color(color)

    plt.tight_layout()
    return fig

def block(x, offset, dt=10, us_factor=50, ax=None, alpha=1.0,
                                          color='lightgrey'):
    """block plot."""

    # upsample the stimulus
    time = np.arange(x.size) * dt
    time_us = np.linspace(0, time[-1], us_factor * time.size)
    x = interp1d(time, x, kind='nearest')(time_us)
    maxval, minval = x.max(), x.min()

    ax = plt.gca() if ax is None else ax
    ax.fill_between(time_us, np.ones_like(x) * offset, x+offset, color=color, interpolate=True)
    ax.plot(time_us, x+offset, '-', color=color, alpha=alpha)

    return ax

def osr(models):
    """Generates the OSR figure."""
    if isinstance(models,nn.Module):
        models = [models]
    # Generate stimulus.
    dt = 10
    # interval: delay
    # duration: duration of flash
    X, omt_idx = osr_stim(duration=2, interval=6, nflashes=8,
                                                  intensity=-2,
                                                  noise_std=0)

    # Responses
    t = np.linspace(0, len(X)*dt, len(X))
    rn = sqz(models, X)

    # Plot
    plt.clf()
    plt.style.use('deepretina')
    fig = plt.figure(figsize=(15,8))
    s = np.s_[models[0].img_shape[0]:140]
    temp_t = t[s]
    nat = rn.mean(1)[s]
    nat_color = 'lightcoral'
    plt.ylabel("Rate (Hz)", fontsize=40)
    plt.xlabel("Time from Omission (s)", fontsize=40)
    ax = plt.gca()
    ax.plot(temp_t, nat, color=nat_color , label='Natural Scenes',
                                           linewidth=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    offset = np.max(rn.mean(1))
    offset = 11
    #ax.plot(t,X[:,-1,0,0]+offset)
    #plt.title('imgs/temp_intv{}_dur{}.png'.format(i,d))
    block(X[:, -1, 0, 0], offset=offset, dt=dt, us_factor=50,
                                                ax=plt.gca(),
                                                color="black")
    zero = t[omt_idx]
    neg2 = t[omt_idx-20]
    neg4 = t[omt_idx-40]
    pos2 = t[omt_idx+20]
    ticks = [neg4, neg2, zero, pos2]
    plt.xlim([neg4-20,pos2+20])
    plt.ylim([0,offset])
    plt.yticks([0,5,10])
    #plt.locator_params(nbins=3)
    labels = [-0.4, -0.2, 0, .2]
    plt.xticks(ticks=ticks, labels=labels)
    ax.tick_params(axis='both', which='major', labelsize=45)
    l = plt.legend(fontsize=40,loc='upper right',frameon=False,
                                                handletextpad=-2.0, 
                                                handlelength=0,
                                                markerscale=0)
    colors = [nat_color]
    for color,text in zip(colors, l.get_texts()):
        text.set_color(color)

    plt.tight_layout()
    return fig

def retinal_phenomena_figs(model, verbose=True):
    """
    Runs the model through a battering of tests and returns figures
    for each.

    model: torch Module
    """
    figs = []
    fig_names = []
    metrics = dict()
    filt_depth = model.img_shape[0]

    try:
        (fig, (ax0,ax1)), X, resp = step_response(model,
                                    filt_depth=filt_depth)
        figs.append(fig)
        fig_names.append("step_response")
        metrics['step_response'] = None
    except Exception as e:
        print("Error in Step Response")
        print(e)
    try:
        fig = osr(model)
        figs.append(fig)
        fig_names.append("osr")
    except Exception as e:
        print("Error in OSR")
        print(e)
    try:
        (fig, (ax0,ax1)), X, resp = reversing_grating(model,
                                        filt_depth=filt_depth)
        figs.append(fig)
        fig_names.append("reversing_grating")
        metrics['reversing_grating'] = None
    except Exception as e:
        print("Error in Reversing Grating")
        print(e)
    try:
        contrasts = [0.1, 0.7]
        (fig, (_)), _, _ = contrast_adaptation(model, contrasts[1],
                                contrasts[0], filt_depth=filt_depth)
        figs.append(fig)
        fig_names.append("contrast_adaptation")
        metrics['contrast_adaptation'] = None
    except Exception as e:
        print("Error in Contrast Adaptation")
        print(e)
    try:
        fig = contrast_fig(model, contrasts, unit_index=0,
                                  nonlinearity_type="bin",
                                  verbose=verbose)
        figs.append(fig)
        fig_names.append("contrast_fig")
        metrics['contrast_fig'] = None
    except Exception as e:
        print("Error in Contrast Fig")
        print(e)
    try:
        fig = dr_motion_reversal(model)
        figs.append(fig)
        fig_names.append("motion_reversal")
        metrics['motion_reversal'] = None
    except Exception as e:
        print("Error in Motion Reversal")
        print(e)
    try:
        tup = motion_anticipation(model, velocity=0.176,
                                         filt_depth=filt_depth)
        (fig, ax) = tup[0]
        figs.append(fig)
        fig_names.append("motion_anticipation")
        metrics['motion_anticipation'] = None
    except Exception as e:
        print("Error in Motion Anticipation")
        print(e)
    try:
        tup = oms_random_differential(model, filt_depth=filt_depth)
        fig, _, _, diff_response, global_response = tup
        figs.append(fig)
        fig_names.append("oms")
        oms_ratios = global_response.mean(0)/diff_response.mean(0)
        metrics['oms'] = oms_ratios
    except Exception as e:
        print("Error in OMS")
        print(e)
    try:
        fig = f2_response(model)
        figs.append(fig)
        fig_names.append("f2_response")
    except Exception as e:
        print("Error in F2 Response")
        print(e)
    try:
        fig = fast_contrast_adaptation(model)
        figs.append(fig)
        fig_names.append("fast_contrast_adaptation")
    except Exception as e:
        print("Error in Fast Contrast Adaptation")
        print(e)
    try:
        fig = latency_encoding(model)
        figs.append(fig)
        fig_names.append("latency_encoding")
    except Exception as e:
        print("Error in Latency Encoding")
        print(e)
    return figs, fig_names, metrics

