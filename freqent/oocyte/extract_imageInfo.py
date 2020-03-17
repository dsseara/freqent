import pims
import pandas as pd
import numpy as np
import h5py
import os
from glob import glob
import argparse
from scipy.ndimage import gaussian_filter


def image_correction(img_raw, sigma=50):
    '''
    helper function to correct for bleaching and illumination gradients
    in images

    Parameters
    ----------
    img_raw : array-like
        numpy array of image to be corrected
    sigma : float (optional)
        size of Gaussian to smooth with when getting the background
        illumination. Defaults to 50

    Results
    -------
    img_corrected : array-like
        numpy array of corrected image

    Example
    -------
    img_corrected = image_correction(img, sigma=35)
    '''
    img = np.array([im - im.mean() for im in img_raw])

    img_sub = img - img.mean(axis=0)

    img_filt = np.array([gaussian_filter(im, sigma=sigma) for im in img_raw]).mean(axis=0)
    img_filt /= img_filt.max()

    img_corrected = np.array([im / img_filt for im in img_sub])

    return img_corrected


parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, nargs='*',
                    help='path to oocyte image files')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='path to output hdf5 files')
parser.add_argument('--parampath', '-p', type=str, default=None,
                    help='path to file with image parameters')
parser.add_argument('--sigma', '-sig', type=float, default=50,
                    help='width of Gaussian blur to use when finding illumination gradient')


args = parser.parse_args()


if args.savepath is None:
    raise ValueError('specify path to save hdf5 output files')
else:
    savepath = args.savepath

params = pd.read_excel(args.parampath)
expts = np.unique([expt[:-3] for expt in params['experiment']])

# if args.exptID is None:
#     # get all available experiments
#     expts = sorted(list(set([expt[:3] for expt in params['experiment']])))
# else:
#     expts = [args.exptID]

for file in args.datapath:
    # Find parameter index and file ID for the file
    param_idx = params.loc[params['experiment'] == file.split(os.path.sep)[-1][:-4]].index[0]
    exptID = expts[np.where([expt in file for expt in expts])[0][0]]

    # get specific file name, i.e. expt_C1 or expt_C2
    fname = file.split(os.path.sep)[-1].split('.')[0]

    # get name of protein imaged in this file
    protein_name = params.loc[params['experiment'] == fname, 'protein'].iloc[0]

    print('Reading {file}'.format(file=fname))
    with h5py.File(os.path.join(savepath, exptID + '.hdf5')) as f:


        if '/images' not in f:
            imgs_group = f.create_group('images')
        else:
            imgs_group = f['images']

        im = pims.TiffStack(file)
        im_array = np.asarray([img for img in im])

        im_corrected = image_correction(im_array, sigma=args.sigma)



        # create dataset in images group with this information
        if '/images/' + protein_name in f:
            del f['images'][protein_name]
        img_dset = imgs_group.create_dataset(protein_name, data=im_corrected)

        # set attributes with metadata
        img_dset.attrs['path'] = file
        img_dset.attrs['sigma'] = args.sigma
        for (mdname, mdcontent) in params.loc[params['experiment'] == fname].iteritems():
            img_dset.attrs[mdname] = mdcontent.iloc[0]



# for expt in expts:
#     if expt == '171007_2':
#         continue
#     print(expt)
#     files = glob(os.path.join(args.datapath, expt + '*'))
#     if len(files) is not 2:
#         Warning('Expecting 2 files, found {n}'.format(n=len(files)))
#         continue

#     with h5py.File(os.path.join(savepath, expt + '.hdf5'), 'w') as f:
#         imgs_group = f.create_group('images')
#         for file in files:
#             # load image data and turn into 3D numpy array
#             im = pims.TiffStack(file)
#             im_array = np.asarray([im[ii] for ii in range(len(im))])

#             # get specific file name, i.e. expt_C1 or expt_C2
#             fname = file.split(os.path.sep)[-1].split('.')[0]

#             # get name of protein imaged in this file
#             protein_name = params.loc[params['experiment'] == fname, 'protein'].iloc[0]

#             # create dataset in images group with this information
#             img_dset = imgs_group.create_dataset(protein_name, data=im_array)

#             # set attributes with metadata
#             img_dset.attrs['path'] = file
#             for (mdname, mdcontent) in params.loc[params['experiment'] == fname].iteritems():
#                 img_dset.attrs[mdname] = mdcontent.iloc[0]
