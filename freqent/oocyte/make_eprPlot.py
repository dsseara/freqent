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


args = parser.parse_args()
file = args.file.split(os.path.sep)[-1].split('.')[0]

with h5py.File(args.file, 'r') as d:
    epf_aziavg = d['entropy']['epf_azi_avg'][:]
    kr = d['entropy']['k_r'][:]
    w = d['entropy']['omega'][:]

fig, ax = plt.subplots()
# ax.pcolormesh(kr, w[w > 0], (epf_aziavg / epf_aziavg.sum(axis=0))[w > 0],
#               rasterized=True, norm=mpl.colors.LogNorm())
a = ax.pcolormesh(kr, w[w > 0], epf_aziavg[w > 0],
                  rasterized=True, norm=mpl.colors.LogNorm(), cmap='magma')
ax.set(xlabel=r'$q_r \ [2 \pi / \mu m]$', ylabel=r'$\omega \ [2 \pi / s]$',
       title=file)

cbar = fig.colorbar(a, ax=ax)
cbar.ax.set(title=r'$\mathcal{E}$')
ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
plt.tight_layout()

if args.savepath is not None:

    fig.savefig(os.path.join(args.savepath, datetime.now().strftime('%y%m%d') + '_' + file + '_epf.pdf'), format='pdf')

plt.show()
