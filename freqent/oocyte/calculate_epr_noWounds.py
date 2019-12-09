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
    print('Reading {f}'.format(f=file.split(os.path.sep)[-1]))
    with h5py.File(file) as d:
        dt = d['images']['actin'].attrs['dt']
        dx = d['images']['actin'].attrs['dx']
        traj = np.stack([d['images']['actin'][:-1], d['images']['rho'][:-1]])
        c, w = fen.corr_matrix(traj, sample_spacing=[dt, dx, dx],
                               window='boxcar', detrend='constant')
        c_azi_avg, w_azi_avg = fen.corr_matrix(traj, sample_spacing=[dt, dx, dx],
                                               window='boxcar', detrend='constant',
                                               azimuthal_average=True)
        s, epf, w = fen.entropy(traj, sample_spacing=[dt, dx, dx],
                                window='boxcar', detrend='constant',
                                smooth_corr=True, nfft=None,
                                sigma=sigma,
                                subtract_bias=True,
                                many_traj=False,
                                return_density=True)
        dw, dkx, dky = [np.diff(k)[0] for k in w]
        epf_azi_avg, kr = fen._azimuthal_average_3D(epf, tdim=0, dx=dkx)

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

expts = ['140706_08',
         '140706_09',
         '140713_08',
         '140713_09',
         '140717_01',
         '140717_13',
         '140817_05',
         '160403_09',
         '160403_14',
         '160915_09',
         '161001_04',
         '161025_01',
         '171230_04']
files = [os.path.join(datapath, expt) for expt in expts]
sigma = [1, 1, 1]
wounds = 0

print('Calculating eprs...')
with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calc_epr_spectral, files)
print('Done.')
