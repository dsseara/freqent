import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse


def divcurltheta(u, v, dx):
    '''
    calculate divergence, curl, and angle of a velocity field

    Parameters
    ----------
    u : array_like
        2D array of x-components of velocity field
    v : array_like
        2D array of y-components of velocity field
    dx : list of scalar or array, optional
        physical spacing between points in velocity field.
        Defaults to 1 for all directions.

    Returns
    -------
    div : array_like
        2D array of velocity divergence field
    curl : array_like
        2D array of velocity curl
    theta : array_like
        2D array of velocity phase angle

    See also
    --------
    np.gradient
    '''
    # if type(dx) is not list:
    #     return TypeError('dx has to be given as a list')

    # # if only passing a scalar, repeat for both axes of u and v
    # if len(dx) == 1:
    #     dx = dx * 2

    dudy, dudx = np.gradient(np.flip(u, axis=0), dx[0], dx[1])
    dvdy, dvdx = np.gradient(np.flip(v, axis=0), dx[0], dx[1])

    div = dudx + dvdy
    curl = dvdx - dudy
    theta = np.arctan2(np.flip(v, axis=0), np.flip(u, axis=0))
    return div, curl, theta


parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, nargs='*',
                    help='absolute path to oocyte hdf5 file(s)')
args = parser.parse_args()

proteins = ['actin', 'rho']
print('Calculating divergence, curl, and phase of...')
for file in args.datapath:
    with h5py.File(file) as d:
        if '/piv' in d:
            print(file.split(os.path.sep)[-1])
            for protein in proteins:
                dx = np.diff(d['piv'][protein]['x'][0])[0]
                div_t = np.zeros(d['piv'][protein]['u'][:].shape)
                curl_t = np.zeros(d['piv'][protein]['u'][:].shape)
                theta_t = np.zeros(d['piv'][protein]['u'][:].shape)

                for tind, (u, v) in enumerate(zip(d['piv'][protein]['u'], d['piv'][protein]['v'])):
                    div_t[tind], curl_t[tind], theta_t[tind] = divcurltheta(u, v, dx=[dx, dx])

                if '/piv/' + protein + '/div' in d:
                    del d['piv'][protein]['div']
                if '/piv/' + protein + '/curl' in d:
                    del d['piv'][protein]['curl']
                if '/piv/' + protein + '/theta' in d:
                    del d['piv'][protein]['theta']
                d['piv'][protein].create_dataset('div', data=div_t)
                d['piv'][protein]['div'].attrs['description'] = 'div of velocity field'
                d['piv'][protein].create_dataset('curl', data=curl_t)
                d['piv'][protein]['curl'].attrs['description'] = 'curl of velocity field'
                d['piv'][protein].create_dataset('theta', data=theta_t)
                d['piv'][protein]['theta'].attrs['description'] = 'theta of velocity field'
        else:
            pass
