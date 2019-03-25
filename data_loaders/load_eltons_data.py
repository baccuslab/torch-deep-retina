import scipy.io as spio
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.signal as ss

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
    size = int(np.round(filt.size / 2))

    # Filter  binned spike times
    return ss.fftconvolve(filt, bspk, mode='full')[size:size + time.size] / dt

def get_eltons_data(spikes_fn,movie_fn,history=20):

    TIME_FIRST_FRAME = 10760. # time when WN starts (n-th system recording frame)
    WN_FRAME_RATE = 1/0.03327082098251457 # frame rate of white noise (1/sec)
    SYS_REP_RATE = 20000. # rep fate of recording system (1/sec)
    KERNEL_FRAMES = history
    WN_LENGTH = 30.*60. # white noise duration (sec)

    cells = spio.loadmat(spikes_fn, struct_as_record=False, squeeze_me=True)
    spikeTimes = dict()

    for n in range(0,cells['spikeTimes'].shape[0]):
        spikes = cells['spikeTimes'][n] - TIME_FIRST_FRAME - KERNEL_FRAMES / WN_FRAME_RATE * SYS_REP_RATE


        spikes = spikes[spikes > 0]
        spikeTimes[n] = np.zeros([round(WN_LENGTH*WN_FRAME_RATE)])

        for st in spikes:
            if np.floor(st/SYS_REP_RATE*WN_FRAME_RATE) >= round(WN_LENGTH*WN_FRAME_RATE): break
            spikeTimes[n][int(np.floor(st/SYS_REP_RATE*WN_FRAME_RATE))] += 1


    print(f"Loaded N={len(spikeTimes)} cells")
    print(f"Data of 1st cell has dimensions {spikeTimes[0].shape}")

    stimulus = []
    with open(movie_fn, newline='') as csvfile:
        wn_movie = csv.reader(csvfile, delimiter=',')
        for row in wn_movie:
            stimulus.append(np.asarray(row))

    stimulus = np.asarray(stimulus)
    stimulus = stimulus.astype('float32')

    num_frames = int(stimulus.shape[0]/20)

    wn_movie = np.zeros([num_frames,20,20])

    for frame in range(num_frames):
        f = []
        for row in range(frame,frame+20):
            f.append(stimulus[row])
        f = np.asarray(f)
        wn_movie[frame,:,:] = f

    stimulus_cube = np.zeros([num_frames - KERNEL_FRAMES, KERNEL_FRAMES, 20, 20])    

    for frame in range(num_frames - KERNEL_FRAMES):
        stimulus_cube[frame,:,:,:] = wn_movie[frame:frame+KERNEL_FRAMES,:,:]

    print(f"Stimulus cube has dimensions {stimulus_cube.shape}")

    for n in range(0,cells['spikeTimes'].shape[0]):
        spikeTimes[n]

    y = []
    for c in range(13):
        st = estfr(spikeTimes[c],np.arange(0,30.*60.,0.0332708209825145),sigma=0.01)
        y.append(st[:stimulus_cube.shape[0]])

    y = np.asarray(y)
    y = y.T
    return stimulus_cube, y