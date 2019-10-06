import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import matplotlib as mpl

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams['xtick.major.width'] = 2
# mpl.rcParams['xtick.minor.width'] = 2
# mpl.rcParams['ytick.major.width'] = 2
# mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    parentFolder = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190910/'
    saveFolder = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brussfield/'
elif sys.platform == 'darwin':
    parentFolder = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/190910/'
    saveFolder = '/Users/Danny/Dropbox/LLM_Danny/freqent/brussfield/'

folders = glob(os.path.join(parentFolder, 'alpha*'))
alphas = np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders])

epr_spectral = np.zeros((len(alphas), 10))
epr_blind = np.zeros(len(alphas))
epr = np.zeros(len(alphas))

for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        epr[fInd] = d['data']['epr'][()]
        epr_blind[fInd] = d['data']['epr_blind'][()]
        epr_spectral[fInd] = d['data']['s'][:]

        if fInd == 0:
            lCompartment = d['params']['lCompartment'][()]
            nCompartments = d['params']['nCompartments'][()]
            T = d['data']['t_points'][:].max()
            dw = np.diff(d['data']['omega'][:])[0]
            dk = np.diff(d['data']['k'][:])[0]
            k_max = d['data']['k'][:].max()
            w_max = d['data']['omega'][:].max()
            sigma = d['params']['sigma'][:]

V = lCompartment * nCompartments
nrep = 1
nvar = 2
sigma_w = sigma[0] * dw
sigma_k = sigma[1] * dk
bias = (1 / nrep) * (nvar * (nvar - 1) / 2) * (w_max / (T * sigma_w * np.sqrt(np.pi))) * (k_max / (V * sigma_k * np.sqrt(np.pi)))

fig, ax = plt.subplots(figsize=(7, 6.5))
#ax.plot(alphas, epr / V, 'o', label='epr')
#ax.plot(np.sort(alphas, epr_blind / V, 'o', label='epr_blind')
#ax.errorbar(alphas, np.mean(epr_spectral, axis=1) - bias, yerr=np.std(epr_spectral, axis=1), fmt='ko', label='epr_spectral', capsize=5)

ax.plot(alphas[alphas < 30], np.mean(epr_spectral, axis=1)[alphas < 30] - bias, 'ko', label=r'$\dot{S}_{spectral}$')
ax.fill_between(alphas[np.argsort(alphas)[np.sort(alphas) < 30]],
                np.mean(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]] - bias + np.std(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]],
                np.mean(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]] - bias - np.std(epr_spectral, axis=1)[np.argsort(alphas)[np.sort(alphas) < 30]],
                color='k', alpha=0.5)
ax.plot(alphas[np.argsort(alphas)[np.sort(alphas) < 30]] , epr_blind[np.argsort(alphas)[np.sort(alphas) < 30]] / V, 'r', lw=3, label=r'$\dot{S}_{thry}$')

# ax.plot(np.log(np.repeat(np.sort(alphas), 10)), np.ravel(epr_spectral[np.argsort(alphas), :]), 'k.')

ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$', xlim=(0.027, 35))
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.tick_params(which='both', direction='in')
ax.set(yscale='log', xscale='log')
ax.legend()
plt.tight_layout()

#fig.savefig(os.path.join(saveFolder, datetime.now().strftime('%y%m%d') + '_eprPlot_doubleBias.pdf'), format='pdf')
plt.show()
