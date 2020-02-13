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
parser.add_argument('-savepath', '-s', type=str, default=None,
                    help='full path to save location of plots')
parser.add_argument('--frames', '-f', type=int, nargs=2, default=[0, -1],
                    help='beginning and end frames to analyze, 0 indexed')
args = parser.parse_args()

shift = 5


def dampedOscillator(t, A, phi, tau, omega, k):
    return A * np.exp(-t / tau) * np.cos(omega * t + phi) + k


tau = np.zeros(len(args.datapath))
for fInd, file in enumerate(args.datapath):
    fig, ax = plt.subplots()
    with h5py.File(file) as d:
        dt = d['images']['actin'].attrs['dt']
        nt, nx, ny = d['images']['actin'][args.frames[0]:args.frames[1]].shape
        t = np.arange(-nt // 2, nt // 2) * dt
        xcorr = np.zeros(nt)
        start_pixel, end_pixel = int(nx // 2) - shift, int(nx // 2) + shift
        for a, r in zip(d['images']['actin'][args.frames[0]:args.frames[1], start_pixel:end_pixel, :].mean(axis=1).T, d['images'] ['rho'][args.frames[0]:args.frames[1], start_pixel:end_pixel, :].mean(axis=1).T):
            xc = np.correlate(a - a.mean(), r - r.mean(), mode='same')
            xcorr += xc
            ax.plot(t[t >= 0], xc[t >= 0], 'k-', alpha=0.1, lw=0.5)

    xcorr /= ny
    ax.plot(t[t >= 0], xcorr[t >= 0], 'r-', lw=2)

    # # do fitting
    # popt, pcov = optimize.curve_fit(dampedOscillator, t[np.logical_and(t >= 0, t < 500)], xcorr[np.logical_and(t >= 0, t < 500)],
    #                                 p0=[xcorr[t >= 0].max(), -np.pi / 4, 200, 100, 0], method='dogbox')

    # ax.plot(t[np.logical_and(t >= 0, t < 500)], dampedOscillator(t[np.logical_and(t >= 0, t < 500)], *popt),
    #         'b--', label=r'$A e^{-t/\tau} cos(\omega t + \phi) + k$')


    # perform interpolation to find peak
    xcorr_interp = interpolate.interp1d(t[t >= 0], xcorr[t >= 0], kind='cubic')
    t_interp = np.linspace(0, 500, 1001)
    peak_inds = signal.find_peaks(xcorr_interp(t_interp))[0]
    ymin, ymax = ax.set_ylim()
    if len(peak_inds) > 0:
        ax.plot([t_interp[peak_inds[0]], t_interp[peak_inds[0]]], [ymin, ymax], 'k--', label=r'$T = {t:0.1f}$'.format(t=t_interp[peak_inds[0]]))

    ax.set(xlabel=r'$\tau$',
           ylabel=r'$\langle \delta I_{actin}(t) \delta I_{Rho}(t + \tau)]) \rangle_t$',
           title=file.split(os.path.sep)[-1].split('.')[0],
           xlim=[0, 500])
    ax.legend()
    ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
    plt.tight_layout()
    if args.savepath is not None:
        today = datetime.now().strftime('%y%m%d')
        f = file.split(os.path.sep)[-1].split('.')[0]
        fig.savefig(os.path.join(args.savepath, today + '_' + f + '_' + 'xcorr.pdf'), format='pdf')


plt.show()
