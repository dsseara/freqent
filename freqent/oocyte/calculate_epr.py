import os
import sys
from glob import glob
import numpy as np
import h5py
import freqent.freqentn as fen
import multiprocessing
import argparse


def calc_epr_spectral(file):
    '''
    function to pass to multiprocessing pool to calculate epr in parallel
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-1]))

    with h5py.File(file) as d:
        dt = d['images']['actin'].attrs['dt']
        dx = d['images']['actin'].attrs['dx']

        traj = np.stack([d['images']['actin'][start_frame:end_frame], d['images']['rho'][start_frame:end_frame]])

        # calculate dynamic structure factor and it's azimuthal average
        c, w = fen.corr_matrix(traj, sample_spacing=[dt, dx, dx],
                               window=window, detrend='constant')

        c_azi_avg, w_azi_avg = fen.corr_matrix(traj, sample_spacing=[dt, dx, dx],
                                               window=window, detrend='constant',
                                               azimuthal_average=True)

        if '/dsf' in d:
            del d['dsf']
        dsf_group = d.create_group('dsf')
        dsf_group.attrs['description'] = 'dynamic structure factor calculations for images'
        dsf_group.attrs['window'] = window
        dsf_group.create_dataset('c', data=c)
        dsf_group.create_dataset('omega', data=w[0])
        dsf_group.create_dataset('k_x', data=w[1])
        dsf_group.create_dataset('k_y', data=w[2])
        dsf_group.create_dataset('c_azi_avg', data=c_azi_avg)
        dsf_group.create_dataset('k_r', data=w_azi_avg[1])

        del c, w, c_azi_avg, w_azi_avg

        # calculate entropy, and epf and epf's azimuthal average
        s, epf, w = fen.entropy(traj, sample_spacing=[dt, dx, dx],
                                window=window, detrend='constant',
                                smooth_corr=True, nfft=None,
                                sigma=sigma,
                                subtract_bias=True,
                                many_traj=False,
                                return_epf=True)

        dw, dkx, dky = [np.diff(k)[0] for k in w]

        epf_azi_avg, kr = fen._azimuthal_average_3D(epf, tdim=0,
                                                    center=None,
                                                    binsize=1,
                                                    mask='circle',
                                                    weight=None,
                                                    dx=dkx)

        if '/entropy' in d:
            del d['entropy']
        entropy_group = d.create_group('entropy')
        entropy_group.attrs['description'] = 'entropy calculations for images'
        entropy_group.attrs['sigma'] = sigma
        entropy_group.attrs['window'] = window
        entropy_group.attrs['frames'] = [start_frame, end_frame]
        entropy_group.create_dataset('s', data=s)
        entropy_group.create_dataset('epf', data=epf)
        entropy_group.create_dataset('epf_azi_avg', data=epf_azi_avg)
        entropy_group.create_dataset('omega', data=w[0])
        entropy_group.create_dataset('k_x', data=w[1])
        entropy_group.create_dataset('k_y', data=w[2])
        entropy_group.create_dataset('k_r', data=kr)

    return s, epf, w


parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, nargs='*',
                    help='full path to data to calculate epr')
parser.add_argument('--frames', '-f', type=int, nargs=2, default=[0, -1],
                    help='beginning and end frames to analyze, 0 indexed')
parser.add_argument('--sigma', '-s', type=float, nargs=3, default=[1, 1, 1],
                    help='smoothing width of Gaussian in t, x, y direction')
parser.add_argument('--window', '-w', type=str, default='boxcar',
                    help='window function to apply to data when finding cross-spectral density')
args = parser.parse_args()
files = list(args.datapath)
start_frame, end_frame = args.frames
sigma = args.sigma
window = args.window

print('Calculating eprs...')
if len(files) > 1:
    with multiprocessing.Pool(processes=2) as pool:
        result = pool.map(calc_epr_spectral, files)
else:
    s, epf, w = calc_epr_spectral(files[0])
print('Done.')

# these are the list of experiments that don't have any underlying problems
# in the images
# expts = ['140706_08',
#          '140706_09',
#          '140713_08',
#          '140713_09',
#          '140717_01',
#          '140717_13',
#          '140817_05',
#          '160403_09',
#          '160403_14',
#          '160915_09',
#          '161001_04',
#          '161025_01',
#          '171230_04']
# files = [os.path.join(datapath, expt + '.hdf5') for expt in expts]
# sigma = [0.01, 0.1, 0.1]
# window = 'hann'

# print('Calculating eprs...')
# with multiprocessing.Pool(processes=2) as pool:
#     result = pool.map(calc_epr_spectral, files)
# print('Done.')
