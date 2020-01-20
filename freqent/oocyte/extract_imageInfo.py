import pims
import pandas as pd
import numpy as np
import h5py
import os
from glob import glob
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', '-d', type=str,
                    help='path to oocyte image files')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='path to output hdf5 files')
parser.add_argument('--exptID', '-id', type=str, default=None,
                    help='If only want specific experiment, put its ID (i.e. 140706_08)')

args = parser.parse_args()

datapath = args.datapath
if args.savepath is None:
    savepath = datapath
else:
    savepath = args.savepath

params = pd.read_excel(os.path.join(datapath, '_params.xlsx'))
if args.exptID is None:
    # get all available experiments
    expts = sorted(list(set([expt[:3] for expt in params['experiment']])))
else:
    expts = [args.exptID]

for expt in expts:
    if expt == '171007_2':
        continue
    print(expt)
    files = glob(os.path.join(datapath, expt + '*'))
    if len(files) is not 2:
        Warning('Expecting 2 files, found {n}'.format(n=len(files)))
        continue

    with h5py.File(os.path.join(savepath, expt + '.hdf5'), 'w') as f:
        imgs_group = f.create_group('images')
        for file in files:
            # load image data and turn into 3D numpy array
            im = pims.TiffStack(file)
            im_array = np.asarray([im[ii] for ii in range(len(im))])

            # get specific file name, i.e. expt_C1 or expt_C2
            fname = file.split(os.path.sep)[-1].split('.')[0]

            # get name of protein imaged in this file
            protein_name = params.loc[params['experiment'] == fname, 'protein'].iloc[0]

            # create dataset in images group with this information
            img_dset = imgs_group.create_dataset(protein_name, data=im_array)

            # set attributes with metadata
            img_dset.attrs['path'] = file
            for (mdname, mdcontent) in params.loc[params['experiment'] == fname].iteritems():
                img_dset.attrs[mdname] = mdcontent.iloc[0]
