import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

plt.rcParams["font.family"] = "Cambria"
plt.rcParams['font.size'] = 10

def multi_hist(path1, path2):
    dataframe = pd.read_csv(path1, header=None)
    DRG = dataframe.values
    RMS = np.sqrt(np.mean(DRG ** 2, axis=1))
    VAR = np.var(DRG, axis=1)
    SKEW = scipy.stats.skew(DRG, axis=1)

    (f, Pa) = scipy.signal.welch(DRG, 25000, window='hann', scaling='spectrum', axis=1)
    Pa = 10 * np.log10(Pa)
    min_val = tf.reduce_min(Pa)
    max_val = tf.reduce_max(Pa)
    Pa = (Pa - min_val) / (max_val - min_val)
    Pa = np.array(Pa)
    AMP = np.max(Pa[:, (f > 2000)], axis=1)# & (f < 1200)

    dataframe2 = pd.read_csv(path2, header=None)
    DRG2 = dataframe2.values
    RMS2 = np.sqrt(np.mean(DRG2 ** 2, axis=1))
    VAR2 = np.var(DRG2, axis=1)
    SKEW2 = scipy.stats.skew(DRG2, axis=1)

    (f2, Pa2) = scipy.signal.welch(DRG2, 25000, window='hann', scaling='spectrum', axis=1)
    Pa2 = 10 * np.log10(Pa2)
    min_val2 = tf.reduce_min(Pa2)
    max_val2 = tf.reduce_max(Pa2)
    Pa2 = (Pa2 - min_val2) / (max_val2 - min_val2)
    Pa2 = np.array(Pa2)
    AMP2 = np.max(Pa2[:, (f > 2000)], axis=1)# & (f < 1200)

    rms = [RMS, RMS2]
    amp = [AMP, AMP2]
    var = [VAR, VAR2]
    skew = [SKEW, SKEW2]

    color = ['blue', 'red']
    labels = ['Gear pair I', 'Gear pair II']

    bin = 20
    fig, ax = plt.subplots(2, 2)
    #fig.set_figheight(2.4)

    ax[0,0].hist(rms, bins=bin, histtype='bar', color=color, label=labels)
    ax[0,0].set_xlabel('Root mean square, $\mathdefault{m/s^2}$')
    ax[0,0].set_ylabel('Density')
    ax[0,0].set_title('a)')
    #ax[0,0].grid(visible=True)
    ax[0,0].legend(prop={'size': 10})

    ax[0,1].hist(amp, bins=bin, color=color, label=labels)
    ax[0,1].set_xlabel('Normalized amplitude')
    ax[0,1].set_ylabel('Density')
    ax[0,1].set_title('b)')
    #ax[0,1].grid(visible=True)
    ax[0,1].legend(prop={'size': 10})

    ax[1,0].hist(var, bins=bin, color=color, label=labels)
    ax[1,0].set_xlabel('Variance, $\mathdefault{(m/s^2)^2}$')
    ax[1,0].set_ylabel('Density')
    ax[1,0].set_title('c)')
    #ax[1,0].grid(visible=True)
    ax[1,0].legend(prop={'size': 10})

    ax[1,1].hist(skew, bins=bin, color=color, label=labels)
    ax[1,1].set_xlabel('Skewness')
    ax[1,1].set_ylabel('Density')
    ax[1,1].set_title('d)')
    #ax[1,1].grid(visible=True)
    ax[1,1].legend(prop={'size': 10})

    fig.tight_layout()
    plt.show()

path_good = 'vibr/good/DRG.csv'
path_fail = 'vibr/fail/DRG_fail.csv'

multi_hist(path_good, path_fail)

drg = pd.read_excel('vibr/good/vibrations (1).xlsx', sheet_name=2, usecols='B', skiprows=7)
t = pd.read_excel('vibr/good/vibrations (1).xlsx', sheet_name=2, usecols='A', skiprows=7)
t = np.array(t)
dt = np.diff(t, axis=0)

N = len(t)
fp = N
print('fp:', fp)
print('N:', N)

dataframe = pd.read_csv(path_good, header=None)
DRG = dataframe.values
(f, Pa) = scipy.signal.welch(DRG, fp, window='hann', scaling='spectrum', axis=1)
Pa_log = 10 * np.log10(Pa)

min_val = tf.reduce_min(Pa_log)
max_val = tf.reduce_max(Pa_log)

Pa_norm = (Pa_log - min_val) / (max_val - min_val)
Pa_norm = np.array(Pa_norm)

no = 1

fig, axs = plt.subplots(ncols=2, nrows=3)
gs = axs[1, 0].get_gridspec()

for ax in axs[0, 0:]:
    ax.remove()
axbig = fig.add_subplot(gs[0, 0:])
axbig.plot(t, drg)
axbig.set_xlabel('time, s')
axbig.set_ylabel('a, $\mathdefault{m/s^2}$')
axbig.set_title('a)')
axbig.grid(visible=True)
axbig.autoscale(enable=True, axis='x', tight=True)

axs[1, 0].plot(np.linspace(0, 1 / 40, 625), DRG[no, :])
axs[1, 0].set_xlabel('time, s')
axs[1, 0].set_ylabel('a, $\mathdefault{m/s^2}$')
axs[1, 0].set_title('b)')
axs[1, 0].grid(visible=True)
axs[1, 0].autoscale(enable=True, axis='x', tight=True)

axs[1, 1].plot(f, Pa[no, :])
axs[1, 1].set_xlabel('frequency, Hz')
axs[1, 1].set_ylabel('$\mathdefault{P_{aa}}$, $\mathdefault{(m/s^2)^2}$')
axs[1, 1].set_title('c)')
axs[1, 1].grid(visible=True)
axs[1, 1].autoscale(enable=True, axis='x', tight=True)

axs[2, 0].plot(f, Pa_log[no, :])
axs[2, 0].set_xlabel('frequency, Hz')
axs[2, 0].set_ylabel('10log($\mathdefault{P_{aa}}$), dB')
axs[2, 0].set_title('d)')
axs[2, 0].grid(visible=True)
axs[2, 0].autoscale(enable=True, axis='x', tight=True)

axs[2, 1].plot(f, Pa_norm[no, :])
axs[2, 1].set_xlabel('frequency, Hz')
axs[2, 1].set_ylabel('x')
axs[2, 1].set_title('e)')
axs[2, 1].grid(visible=True)
axs[2, 1].autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()

plt.show()

