import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
import freqent.freqentn as fen

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190910'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/190910'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity'

# alpha = 65.24
files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))
sigma = [20, 5]

for file in files:
    with h5py.File(file, 'r') as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 10)[0]
        dt = np.diff(t_points)[0]
        dx = d['params']['lCompartment'][()]
        # nCompartments = d['params']['nCompartments'][()]
        nSim = d['params']['nSim'][()]
        alpha = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                 (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))

        s = np.zeros(nSim)
        nt, nx = d['data']['trajs'][0, 0, t_epr, :].shape
        rhos = np.zeros((nSim, nt - (nt + 1) % 2, nx - (nx + 1) % 2))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr, :]):
            s[ind], rhos[ind], w = fen.entropy(traj, sample_spacing=[dt, dx],
                                               window='boxcar', detrend='constant',
                                               smooth_corr=True, nfft=None,
                                               sigma=sigma,
                                               subtract_bias=True,
                                               many_traj=False,
                                               return_density=True)

    fig, ax = plt.subplots()
    a = ax.pcolormesh(w[1] * 100, w[0], rhos.mean(axis=0), rasterized=True)
    ax.set(xlabel=r'$k$', ylabel=r'$\omega$',
           ylim=[-20, 20])
    ax.tick_params(which='both', direction='in')

    cbar = fig.colorbar(a, ax=ax)
    cbar.ax.tick_params(which='both', direction='in')
    cbar.ax.set(title=r'$\rho_{\dot{s}}$')
    ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
    plt.tight_layout()

    fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_eprDensity_alpha{a}.pdf'.format(a=alpha)), format='pdf')

    plt.close('all')

