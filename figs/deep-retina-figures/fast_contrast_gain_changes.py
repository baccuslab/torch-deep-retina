import matplotlib
matplotlib.use('Agg')

import numpy as np
import deepdish as dd
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import h5py
plt.style.use('deepretina')

prepath = "convgc_"
gain_changes = dd.io.load(prepath+'fast_contrast_gain_changes.h5')
contrasts = [0.5, 1.0, 1.5, 2.0]

# Create figure for slope of nonlinearity vs stimulus contrast.
plt.figure(figsize=(5, 3))
plt.errorbar(contrasts, [np.mean(gain_changes['naturalscene'][contrast]) for contrast in contrasts],
             yerr=[sem(gain_changes['naturalscene'][contrast]) for contrast in contrasts],
             fmt='o', color='#F07F7F', label='natural scenes')
plt.errorbar(contrasts, [np.mean(gain_changes['whitenoise'][contrast]) for contrast in contrasts], 
             yerr=[sem(gain_changes['whitenoise'][contrast]) for contrast in contrasts],
             fmt='o', color='#7F7F7F', label='white noise')
plt.xlabel('Contrast', fontsize=20)
plt.ylabel('Slope (Hz/Filtered Input)', fontsize=20)

ax1 = plt.gca()
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
majorLocator = MultipleLocator(10)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(0.5)
ax1.xaxis.set_major_locator(minorLocator)
ax1.yaxis.set_major_locator(majorLocator)
ax1.yaxis.set_major_formatter(majorFormatter)

plt.savefig(prepath+'fast_contrast_gain_changes.png', dpi=200)

