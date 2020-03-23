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
parser.add_argument('--save', '-s', type=bool, default=False,
                    help='boolean of whether to save xcorr and params')
parser.add_argument('--frames', '-f', type=int, nargs=2, default=[0, -1],
                    help='beginning and end frames to analyze, 0 indexed')
parser.add_argument('--fit', type=str, default='expDecay',
                    help='which fit to use, either dampedOscillator or expDecay (only to peaks)')
args = parser.parse_args()

shift = 5


def dampedOscillator(t, A, tau, omega, phi, k):
    return A * np.exp(-t / tau) * np.cos(omega * t + phi) + k


def expDecay(t, A, tau, k):
    '''
    Only use if full damped harmonic oscillator fit doesn't work
    use on peaks of oscillatory peaks
    '''
    return A * np.exp(-t / tau) + k


tau = np.zeros(len(args.datapath))
for fInd, file in enumerate(args.datapath):
    with h5py.File(file) as d:
        dt = d['images']['actin'].attrs['dt']
        nt, nx, ny = d['images']['actin'][args.frames[0]:args.frames[1]].shape
        t = np.arange(-nt, nt - 1) * dt
        xcorr = np.zeros((2 * nt - 1, ny))
        start_pixel, end_pixel = int(nx // 2) - shift, int(nx // 2) + shift
        start_frame, end_frame = args.frames[0], args.frames[1]
        for ind, (a, r) in enumerate(zip(d['images']['actin'][start_frame:end_frame, start_pixel:end_pixel, :].mean(axis=1).T,
                                         d['images']['rho'][start_frame:end_frame, start_pixel:end_pixel, :].mean(axis=1).T)):
            xcorr[:, ind] = np.correlate(a - a.mean(), r - r.mean(), mode='full')

    # do fitting
    if args.fit == 'dampedOscillator':
        popt, pcov = optimize.curve_fit(dampedOscillator,
                                        t[t > 0],
                                        xcorr.mean(axis=1)[t > 0],
                                        p0=[1e7, 200, 0.06, 0, -8e4], method='dogbox')

    elif args.fit == 'expDecay':
        # perform interpolation to find peak fo mean xcorr
        xcorr_interp = interpolate.interp1d(t[t >= 0], xcorr.mean(axis=1)[t >= 0], kind='cubic')
        t_interp = np.linspace(0, t.max(), 10001)
        peak_inds = signal.find_peaks(xcorr_interp(t_interp))[0]
        popt, pcov = optimize.curve_fit(expDecay,
                                        t_interp[peak_inds],
                                        xcorr_interp(t_interp[peak_inds]),
                                        p0=[1e7, 100, -1e5], method='dogbox')
    else:
        raise ValueError('Unrecognized fit function. Either dampedOscillator or expDecay.')

    if args.save:
        with h5py.File(file) as d:
            if '/xcorr' in d:
                del d['xcorr']
            xcorr_group = d.create_group('xcorr')

            if args.fit == 'dampedOscillator':
                fit = 'damped harmonic oscillator, A exp(-t/tau) cos(omega t + phi) + k'
            elif args.fit == 'expDecay':
                fit = 'exponential decay, A exp(-t/tau) + k'

            description = 'cross correlation and between actin and rho, plus fit'

            xcorr_group.attrs['description'] = description
            xcorr_group.attrs['fit'] = fit
            xcorr_group.create_dataset('xcorr_mean', data=xcorr)
            xcorr_group.create_dataset('t', data=t)

            if args.fit == 'dampedOscillator':
                xcorr_group.create_dataset('A', data=popt[0])
                xcorr_group.create_dataset('A_std', data=np.sqrt(pcov[0, 0]))
                xcorr_group.create_dataset('tau', data=popt[1])
                xcorr_group.create_dataset('tau_std', data=np.sqrt(pcov[1, 1]))
                xcorr_group.create_dataset('omega', data=popt[2])
                xcorr_group.create_dataset('omega_std', data=np.sqrt(pcov[2, 2]))
                xcorr_group.create_dataset('phi', data=popt[3])
                xcorr_group.create_dataset('phi_std', data=np.sqrt(pcov[3, 3]))
                xcorr_group.create_dataset('k', data=popt[4])
                xcorr_group.create_dataset('k_std', data=np.sqrt(pcov[4, 4]))
            elif args.fit == 'expDecay':
                xcorr_group.create_dataset('A', data=popt[0])
                xcorr_group.create_dataset('A_std', data=np.sqrt(pcov[0, 0]))
                xcorr_group.create_dataset('tau', data=popt[1])
                xcorr_group.create_dataset('tau_std', data=np.sqrt(pcov[1, 1]))
                xcorr_group.create_dataset('k', data=popt[2])
                xcorr_group.create_dataset('k_std', data=np.sqrt(pcov[2, 2]))
                xcorr_group.create_dataset('omega', data=(2 * np.pi / np.diff(t_interp[peak_inds])).mean())
