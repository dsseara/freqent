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
parser.add_argument('datapath', type=str,
                    help='full path to hdf5 file with g(r) data')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='where to save output plot')

args = parser.parse_args()
file = args.datapath.split(os.path.sep)[-1].split('.')[0]

with h5py.File(args.datapath, 'r') as d:
    gr = d['granules']['pair_corr']['gr'][:]
    r = d['granules']['pair_corr']['r'][:]
    dx = d['images']['actin'].attrs['dx']


fig, ax = plt.subplots()
ax.plot(r[:-1] * dx, gr.T, 'k', alpha=0.1, lw=0.5)
ax.plot(r[:-1] * dx, gr.mean(axis=0), 'r', lw=2)
# ax.plot(r[:-1] * dx, [1] * len(r[:-1]), 'w--', alpha=0.75)
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.set(xlabel=r'$r \ (\mu m)$', ylabel=r'$g(r)$')
plt.tight_layout()

if args.savepath is not None:
    today = datetime.now().strftime('%y%m%d')
    fig.savefig(os.path.join(args.savepath, today + '_' + file + '_gr.pdf'), format='pdf')

plt.show()
