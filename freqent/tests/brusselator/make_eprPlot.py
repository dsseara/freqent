import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime

plt.close('all')
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams['xtick.major.width'] = 2
# mpl.rcParams['xtick.minor.width'] = 2
# mpl.rcParams['ytick.major.width'] = 2
# mpl.rcParams['ytick.minor.width'] = 2

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

fig, ax = plt.subplots(figsize=(7, 6.5))
# ax.plot(alphas, epr, 'o', label='epr')
# ax.errorbar(alphas[alphas < 30], np.mean(epr_spectral, axis=1)[alphas < 30],
#             yerr=np.std(epr_spectral, axis=1)[alphas < 30], fmt='ko',
#             label=r'$\dot{S}_{spectral}$', capsize=5)
ax.plot(alphas[alphas < 30], np.mean(epr_spectral, axis=1)[alphas < 30], 'ko', label=r'$\dot{S}_{spectral}$')
ax.fill_between(alphas[np.argsort(alphas)[np.sort(alphas) < 30]],
                np.mean(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]] + np.std(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]],
                np.mean(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]] - np.std(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]],
                color='k', alpha=0.5)
ax.plot(alphas[np.argsort(alphas)[np.sort(alphas) < 30]] , epr_blind[np.argsort(alphas)[np.sort(alphas) < 30]], 'r', lw=3, label=r'$\dot{S}_{thry}$')
# ax.plot(np.repeat(np.sort(alphas), 50), np.ravel(epr_spectral[np.argsort(alphas), :]), 'k.', alpha=0.5)

ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$')
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.set(yscale='log', xscale='log')
ax.tick_params(which='both', direction='in')
ax.legend()
plt.tight_layout()

# fig.savefig(os.path.join(saveFolder, datetime.now().strftime('%y%m%d') + '_eprPlot.pdf'), format='pdf')
plt.show()
