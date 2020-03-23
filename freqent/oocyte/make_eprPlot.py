import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import sys
from datetime import datetime
from glob import glob
import pandas as pd
import argparse
import seaborn as sns

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str,
                    help='parent directory of oocyte hdf5 files')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='path to output hdf5 files')

args = parser.parse_args()

datapath = args.datapath
if args.savepath is None:
    savepath = datapath
else:
    savepath = args.savepath
today = datetime.now().strftime('%y%m%d')

files = glob(os.path.join(datapath, '*.hdf5'))
epr = []
excitability = []
f = []
notes_actin = []
notes_rho = []
length = []

for file in files:
    if file == '/mnt/llmStorage203/Danny/oocyte/160915_09.hdf5':
        continue
    with h5py.File(file, 'r') as d:
        if '/entropy' in d:
            f.append(file.split(os.path.sep)[-1].split('.')[0])
            epr.append(d['entropy']['s'][()])
            length.append(d['images']['actin'][:].shape[0])
            notes_actin.append(d['images']['actin'].attrs['Notes'])
            notes_rho.append(d['images']['rho'].attrs['Notes'])

            # first two conditions work for bement data. last condition works with michaud data
            if d['images']['actin'].attrs['excitability'] == 'ctrl':
                excitability.append(-1)
            elif d['images']['actin'].attrs['excitability'] == 'Ect2':
                excitability.append(0)
            else:
                excitability.append(d['images']['actin'].attrs['excitability'])

            window = d['entropy'].attrs['window']
            sigma = d['entropy'].attrs['sigma']
        else:
            pass

epr_df = pd.DataFrame({'epr': epr, 'excitability': excitability, 'file': f,
                       'length': length, 'notes_actin': notes_actin})
inds = np.logical_and(['ok' in note for note in epr_df['note_actin']],
                      ['ok' in note for note in epr_df['note_rho']],
                      epr_df['len'] >= 50)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(epr_df[inds]['excitability'], epr_df[inds]['epr'], 'ko')
ax.set(xlabel=r'[ArhGAP] (ng/$\mu$l', ylabel=r'$ds/dt$')
# sns.despine(offset={'left': -30}, trim=True)
plt.tight_layout()

if args.savepath is not None:
    fig.savefig(os.path.join(savepath, today + '_epr_vs_arhgap.pdf'.format(window=window, *sigma)))
plt.show()
