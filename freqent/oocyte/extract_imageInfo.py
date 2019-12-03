import pims
import pandas as pd
import numpy as np
import h5py
import os
from glob import glob
import sys

if sys.platform == 'darwin':
    datapath = '/Users/Danny/Dropbox/Excitable wounds for Mike and Ian/'
if sys.platform == 'linux':
    datapath = '/media/daniel/storage11/Dropbox/Excitable wounds for Mike and Ian/'

params = pd.read_excel(os.path.join(datapath, '_params.xlsx'))
expts = list(set([expt[:-3] for expt in params['experiment']]))

for expt in expts:
