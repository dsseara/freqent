import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from datetime import datetime


plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, nargs='*',
                    help='full path to datafiles')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='full path to save location of plot')
args = parser.parse_args()

for fInd, file in enumerate(args.datapath):
    fig, ax = plt.subplots()
    with h5py.File(file) as d:
        dt = d['images']['actin'].attrs['dt']
        t = d['xcorr']['t'][:]
        xcorr = d['xcorr']['xcorr_mean'][:]
        A, tau, k = d['xcorr']['A'][()], d['xcorr']['tau'][()], d['xcorr']['k'][()]
        A_std, tau_std, k_std = d['xcorr']['A_std'][()], d['xcorr']['tau_std'][()], d['xcorr']['k_std'][()]
    ax.plot(t[t >= 0], xcorr[t >= 0], 'k-', alpha=0.1, lw=0.5)
    ax.plot(t[t >= 0], xcorr.mean(axis=1)[t >= 0], 'r-', lw=2)
    ax.plot(t[t >= 0], A * np.exp(-t[t >= 0] / tau) + k, '--',
            color='cyan', lw=2, label=r'$\tau = {t:0.2f} \pm {t_sig}$'.format(t=tau, t_sig=tau_std))

    ax.set(xlabel=r'$\tau$',
           ylabel=r'$\langle \delta I_{actin}(t) \delta I_{Rho}(t + \tau) \rangle_t$',
           title=file.split(os.path.sep)[-1].split('.')[0])
    ax.legend()
    ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
    plt.tight_layout()
    if args.savepath is not None:
        today = datetime.now().strftime('%y%m%d')
        f = file.split(os.path.sep)[-1].split('.')[0]
        fig.savefig(os.path.join(args.savepath, today + '_' + f + '_' + 'xcorr.pdf'), format='pdf')


plt.show()
