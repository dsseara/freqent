import os
import sys
from glob import glob
import numpy as np
import h5py
import freqent.freqentn as fen
import multiprocessing


def calc_epr_spectral(file):
    '''
    function to pass to multiprocessing pool to calculate epr in parallel
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-2]))
    with h5py.File(file) as d:
        if d['images']['actin'].attrs['woundBool'] == wounds:
            print('has wound, skipping...')
            continue

        dt = d['images']['actin'].attrs['dt']
        dx = d['images']['actin'].attrs['dx']

        traj = np.con

        s, epf, w = fen.entropy(traj, sample_spacing=[dt, dx, dx],
                                      window='boxcar', detrend='constant',
                                      smooth_corr=True, nfft=None,
                                      sigma=sigma,
                                      subtract_bias=True,
                                      many_traj=False,
                                      return_density=True)

        if '/data/s' in d:
            del d['data']['s']
        d['data'].create_dataset('s', data=s)

        if '/data/epf' in d:
            del d['data']['epf']
        d['data'].create_dataset('epf', data=epf)

        if '/data/omega' in d:
            del d['data']['omega']
        d['data'].create_dataset('omega', data=w[0])

        if '/data/k' in d:
            del d['data']['k']
        d['data'].create_dataset('k', data=w[1])

        if '/params/sigma/' in d:
            del d['params']['sigma']
        d['params'].create_dataset('sigma', data=sigma)

    return s, epf, w


if sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/oocyte/'
if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/oocyte/'

files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))
sigma = [1, 1, 1]
wounds = 0

print('Calculating eprs...')
with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calc_epr_spectral, files)
print('Done.')
