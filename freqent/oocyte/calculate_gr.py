import h5py
import os
import numpy as np
import trackpy as tp
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str,
                    help='full path to hdf5 file with granule locations')
parser.add_argument('--cutoff', '-c', type=float, default=100,
                    help='cutoff distance to calculate g(r)')
parser.add_argument('--dr', '-d', type=float, default=1,
                    help='width of bins for g(r)')

args = parser.parse_args()

with h5py.File(args.datapath) as d:
    if '/granules' not in d:
        ValueError('Need granule location data first before running this script')

    if '/granules/pair_corr' in d:
        del d['granules']['pair_corr']

    # reconstruct pandas dataframe with same columns
    # as the output of tp.batch()
    keys = ['ecc', 'ep', 'frame', 'mass', 'raw_mass',
            'signal', 'size', 'x', 'y']
    df = pd.DataFrame()
    for key in keys:
        df[key] = d['granules'][key]

    nbins = int(args.cutoff // args.dr)
    gr = np.zeros((len(df['frame'].unique()), nbins))

    for ind, frame in enumerate(df['frame'].unique()):
        r, gr[ind] = tp.static.pair_correlation_2d(df[df['frame'] == frame],
                                                   cutoff=args.cutoff,
                                                   dr=args.dr)

    pair_corr_group = d['granules'].create_group('pair_corr')
    pair_corr_group.attrs['description'] = 'pair correlation function for granules'
    pair_corr_group.attrs['cutoff'] = args.cutoff
    pair_corr_group.attrs['dr'] = args.dr
    pair_corr_group.create_dataset('gr', data=gr)
    pair_corr_group.create_dataset('r', data=r)
