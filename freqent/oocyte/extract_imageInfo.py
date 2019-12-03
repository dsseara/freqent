import pims
import pandas as pd
import numpy as np
import h5py
import os
from glob import glob
import sys

if sys.platform == 'darwin':
    datapath = '/Users/Danny/Dropbox/Excitable wounds for Mike and Ian/'
    savepath = '/Volumes/Storage/Danny/oocyte'
if sys.platform == 'linux':
    datapath = '/media/daniel/storage11/Dropbox/Excitable wounds for Mike and Ian/'
    savepath = '/mnt/llmStorage203/Danny/oocyte'

params = pd.read_excel(os.path.join(datapath, '_params.xlsx'))
expts = sorted(list(set([expt[:-3] for expt in params['experiment']])))

for expt in expts[:4]:
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
