import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
mpl.rcParams['pdf.fonttype'] = 42
plt.close('all')

if sys.platform == 'linux':
    parentFolder = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190904/'
    saveFolder = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'
elif sys.platform == 'darwin':
    parentFolder = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/190904/'
    saveFolder = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'

folders = glob(os.path.join(parentFolder, 'alpha*'))
alphas = np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders])

epr_spectral = np.zeros((len(alphas), 50))
epr_blind = np.zeros(len(alphas))
epr = np.zeros(len(alphas))

for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        epr[fInd] = d['data']['epr'][()]
        epr_blind[fInd] = d['data']['epr_blind'][()]
        epr_spectral[fInd] = d['data']['s'][:]

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(alphas, epr, 'o', label='epr')
ax.plot(alphas, epr_blind, 'o', label='epr_blind')
ax.errorbar(alphas, np.mean(epr_spectral, axis=1), yerr=np.std(epr_spectral, axis=1), fmt='ko', label='epr_spectral', capsize=5)
# ax.plot(np.repeat(np.sort(alphas), 50), np.ravel(epr_spectral[np.argsort(alphas), :]), 'k.', alpha=0.5)

ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$')
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.set(yscale='log', xscale='log')
ax.tick_params(which='both', direction='in')
plt.legend()

plt.tight_layout()

fig.savefig(os.path.join(saveFolder, datetime.now().strftime('%y%m%d') + '_eprPlot.pdf'), format='pdf')
plt.show()