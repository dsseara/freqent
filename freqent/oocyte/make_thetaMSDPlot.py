import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
import scipy.stats as stats


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


def mean_square_disp(x, dt):
    shifts = np.array(list(range(1, len(x))))
    msd = np.zeros(len(shifts))
    for ind, shift in enumerate(shifts):
        disp = x[:-shift if shift else None] - x[shift:]
        msd[ind] = np.mean(disp**2)
    return msd, shifts * dt


parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, nargs='*',
                    help='full path to datafiles')
parser.add_argument('-savepath', '-s', type=str, default=None,
                    help='full path to save location of plots')
args = parser.parse_args()


# files = ['/mnt/llmStorage203/Danny/oocyte/140706_08.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/140706_09.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/140713_08.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/140717_01.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/140717_13.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/140817_05.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/160403_09.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/160403_14.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/160915_09.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/161001_04.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/161025_01.hdf5',
#          '/mnt/llmStorage203/Danny/oocyte/171230_04.hdf5']

# files = ['/mnt/llmStorage203/Danny/oocyte/160403_14.hdf5']

alpha = np.zeros(len(args.datapath))
for fInd, file in enumerate(args.datapath):
    fig, ax = plt.subplots()
    with h5py.File(file) as d:
        dt = d['images']['actin'].attrs['dt']
        nframes, nvecy, nvecx = d['piv']['actin']['theta'][:, :24, :24].shape
        msd = np.zeros((nvecy * nvecx, nframes - 1))
        for ind, t in enumerate(np.reshape(d['piv']['actin']['theta'][:, :24, :24], (nframes, nvecy * nvecx)).T):
            msd[ind], tau = mean_square_disp(np.unwrap(t), dt)

    alpha[fInd], b, r, p, sigma = stats.linregress(np.log10(tau), np.log10(msd.mean(axis=0)))
    ax.loglog(tau, msd.T, 'k', lw=0.1, alpha=0.1)
    ax.loglog(tau, msd.mean(axis=0), 'r', lw=2, label=r'$\alpha = {a:0.2f}$'.format(a=alpha[fInd]))
    ax.set(xlabel=r'$\tau$',
           ylabel=r'$\langle \vert \theta(t + \tau) - \theta(t) \vert^2 \rangle_t$',
           title=file.split(os.path.sep)[-1].split('.')[0])
    ax.legend()
    plt.tight_layout()


plt.show()
