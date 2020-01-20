import os
import numpy as np
import h5py
import multiprocessing
import argparse
from openpiv import process, validation, filters


def piv_stack(file):
    '''
    function to pass to multiprocessing pool to calculate epr in parallel
    '''
    print(file.split(os.path.sep)[-1])

    with h5py.File(file) as d:

        # get rid of old piv data if rerunning
        if '/piv' in d:
            del d['piv']
        piv_group = d.create_group('piv')

        dt = d['images']['actin'].attrs['dt']
        dx = d['images']['actin'].attrs['dx']

        trajs = np.stack([d['images']['actin'][:-1], d['images']['rho'][:-1]])

        # create list of protein names
        proteins = ['actin', 'rho']

        # get shape of piv fields
        u_shape, v_shape = process.get_field_shape(trajs[0, 0].shape,
                                                   args.winsize,
                                                   args.overlap)

        # create empty piv array in shape of
        # [n_frames - 1, piv_shape_x, piv_shape_y, n_velocity_components]
        uv_array = np.zeros((len(trajs[0]) - 1, u_shape, v_shape, 2))

        for protein_ind, traj in enumerate(trajs):
            for frame_ind, (im0, im1) in enumerate(zip(traj[:-1], traj[1:])):
                # print('Frame {f}'.format(f=frame_ind))
                # get velocity in x and y
                u, v, sig2noise = process.extended_search_area_piv(im0.astype(np.int32), im1.astype(np.int32),
                                                                   window_size=args.winsize,
                                                                   overlap=args.overlap,
                                                                   dt=dt,
                                                                   search_area_size=args.searchsize,
                                                                   sig2noise_method='peak2peak')
                # get rid of all vectors with low signal to noise ratio
                u1, v1, mask = validation.sig2noise_val(u, v, sig2noise,
                                                        threshold=args.threshold)
                # replace all nullified vectors from above with local mean
                u2, v2 = filters.replace_outliers(u1, v1,
                                                  method='localmean',
                                                  max_iter=10,
                                                  kernel_size=2)
                # assign velocity to piv array
                uv_array[frame_ind, :, :, 0] = u2
                uv_array[frame_ind, :, :, 1] = v2

            # get pixel locations of piv vectors. Multiply by dx to get physical locations
            x, y = process.get_coordinates(image_size=trajs[0, 0].shape,
                                           window_size=args.winsize,
                                           overlap=args.overlap)

            if np.isnan(uv_array).any():
                print('{file} {protein} has nans'.format(file=file.split(os.path.sep)[-1], protein=proteins[protein_ind]))

            piv_group.create_group(proteins[protein_ind])
            piv_group[proteins[protein_ind]].create_dataset('u', data=uv_array[..., 0])
            piv_group[proteins[protein_ind]]['u'].attrs['description'] = 'x velocity components over time'
            piv_group[proteins[protein_ind]].create_dataset('v', data=uv_array[..., 1])
            piv_group[proteins[protein_ind]]['v'].attrs['description'] = 'y velocity components over time'
            piv_group[proteins[protein_ind]].create_dataset('x', data=x)
            piv_group[proteins[protein_ind]]['x'].attrs['description'] = 'x coordinates of velocity components, in pixels'
            piv_group[proteins[protein_ind]].create_dataset('y', data=y)
            piv_group[proteins[protein_ind]]['y'].attrs['description'] = 'y coordinates of velocity components, in pixels'

        piv_group.attrs['description'] = 'piv calculations for images'
        piv_group.attrs['window_size'] = args.winsize
        piv_group.attrs['overlap'] = args.overlap
        piv_group.attrs['searchsize'] = args.searchsize
        piv_group.attrs['snr_threshold'] = args.threshold

    return x, y, uv_array


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, nargs='*',
                    help='full path to hdf5 file(s) of image data')
parser.add_argument('--winsize', '-w', type=int, default=24,
                    help='size of window to calculate PIV, in pixels. Defaults to 24')
parser.add_argument('--searchsize', '-s', type=int, default=64,
                    help='size of window to calculate PIV, in pixels. Defaults to 64')
parser.add_argument('--overlap', '-o', type=int, default=None,
                    help='number of pixels that windows overlap. Defaults to half winsize')
parser.add_argument('--threshold', '-t', type=float, default=1.3,
                    help='Signal to noise ratio below which vectors are discarded. Defaults to 1.3')
args = parser.parse_args()

if args.overlap is None:
    args.overlap = args.winsize // 2

print('Performing PIV on...')
if len(args.filename) == 1:
    piv_stack(args.filename[0])
elif len(args.filename) > 1:
    with multiprocessing.Pool(processes=2) as pool:
        result = pool.map(piv_stack, args.filename)
