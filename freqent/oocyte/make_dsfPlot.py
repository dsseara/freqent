import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
from datetime import datetime
import argparse

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
parser.add_argument('--file', '-f', type=str,
                    help='name of hdf5 file containing plot epf')
parser.add_argument('--aziavg', '-aa', type=bool, default=True,
                    help='whether or not to plot the azimuthal average')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='where to save output plot')

args = parser.parse_args()
file = args.file.split(os.path.sep)[-1].split('.')[0]

with h5py.File(args.file, 'r') as d:
    c_aziavg = d['dsf']['c_azi_avg'][:]
    kr = d['dsf']['k_r'][:]
    w = d['dsf']['omega'][:]

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
labels = ['actin', 'rho']
# ax.pcolormesh(kr, w[w > 0], (epf_aziavg / epf_aziavg.sum(axis=0))[w > 0],
#               rasterized=True, norm=mpl.colors.LogNorm())
for axInd, ax in enumerate(axs):
    a = ax.pcolormesh(kr, w[w > 0],
                      (c_aziavg / c_aziavg.sum(axis=0))[w > 0, :, axInd, axInd].real,
                      rasterized=True, norm=mpl.colors.LogNorm(), cmap='viridis')
    ax.set(xlabel=r'$q_r \ [2 \pi / \mu m]$', ylabel=r'$\omega \ [2 \pi / s]$',
           title=file + ', ' + labels[axInd])

    cbar = fig.colorbar(a, ax=ax)
    cbar.ax.set(title=r'$S(q, \omega) / S(q)$')
    ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])

plt.tight_layout()

if args.savepath is not None:
    fig.savefig(os.path.join(args.savepath, datetime.now().strftime('%y%m%d') + '_' + file + '_dsf.pdf'), format='pdf')

plt.show()
