import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import sys
from datetime import datetime
from glob import glob
import pandas as pd
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
parser.add_argument('--datapath', '-d', type=str,
                    help='Absolute path to oocyte hdf5 files')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='Absolute path to save plots. Script creates subfolder there')

args = parser.parse_args()
savefolder = datetime.now().strftime('%y%m%d') + '_' + args.datapath.split(os.path.sep)[-1].split('.')[0] + '_pivOverlay'
savepath = os.path.join(args.savepath, savefolder)

if not os.path.exists(savepath):
    os.mkdir(savepath)

fig, ax = plt.subplots()
with h5py.File(args.datapath) as d:
    dx = d['images']['actin'].attrs['dx']

    # rho piv vectors
    print('plotting rho piv...')
    if not os.path.exists(os.path.join(savepath, 'rho')):
        os.mkdir(os.path.join(savepath, 'rho'))
    for ind, (im, u, v) in enumerate(zip(d['images']['rho'], d['piv']['rho']['u'], d['piv']['rho']['v'])):
        ax.cla()
        ax.imshow(im, rasterized=True, cmap='gist_gray')
        ax.quiver(d['piv']['rho']['x'], np.flip(d['piv']['rho']['y'][:]), u, v,
                  pivot='mid', color='cyan')
        ax.set_aspect('equal')
        fig.savefig(os.path.join(savepath, 'rho', 'frame{f:03d}.png'.format(f=ind)), format='png')

    # actin piv vectors
    print('plotting actin piv...')
    if not os.path.exists(os.path.join(savepath, 'actin')):
        os.mkdir(os.path.join(savepath, 'actin'))
    for ind, (im, u, v) in enumerate(zip(d['images']['actin'], d['piv']['actin']['u'], d['piv']['actin']['v'])):
        ax.cla()
        ax.imshow(im, rasterized=True, cmap='gist_gray')
        ax.quiver(d['piv']['actin']['x'], np.flip(d['piv']['actin']['y'][:]), u, v,
                  pivot='mid', color='red')
        ax.set_aspect('equal')
        fig.savefig(os.path.join(savepath, 'actin', 'frame{f:03d}.png'.format(f=ind)), format='png')

plt.close('all')
