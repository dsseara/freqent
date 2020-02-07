import h5py
import os
import numpy as np
import trackpy as tp
import pandas as pd
import argparse
import pims

parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str,
                    help='full path to image for tracking')
parser.add_argument('savepath', type=str,
                    help='full path to hdf5 file to save output')
parser.add_argument('--diameter', '-d', type=int, default=11,
                    help='diameter, in pixels, of features to locate. Must be odd integer')
parser.add_argument('--minmass', '-min', type=float, default=None,
                    help='minimum size ')
parser.add_argument('--frames', '-f', type=int, nargs=2, default=[0, -1],
                    help='frames over which to locate granules')
parser.add_argument('--invert', '-i', type=bool, default=True,
                    help='Set to True is particles to locate are dark')

args = parser.parse_args()

print('Optimize the parameters used here before running this script!')

frames = pims.open(args.datapath)
f = tp.batch(frames[args.frames[0]:args.frames[1]], args.diameter,
             invert=args.invert,
             minmass=args.minmass)

with h5py.File(args.savepath) as d:
    if '/granules' in d:
        del d['granules']
    granules_group = d.create_group('granules')
    granules_group.attrs['description'] = 'location data for granules'
    granules_group.attrs['file'] = args.datapath.split(os.path.sep)[-1]
    granules_group.attrs['diameter'] = args.diameter
    granules_group.attrs['minmass'] = args.minmass
    granules_group.attrs['frames'] = args.frames
    granules_group.attrs['invert'] = args.invert

    for key in f.keys():
        granules_group.create_dataset(key, data=f[key].values)
