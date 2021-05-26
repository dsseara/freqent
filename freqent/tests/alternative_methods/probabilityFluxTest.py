'''
Test probability flux entropy production method with a simulation of a
Brownian particle in 2D in a harmonic trap and acted upon by a rotational force
dr/dt = F + xi
with
       ---     ---
       | -k   -a |
F(r) = |  a   -k |
       ---     ---
The k terms represents a harmonic potential (set to 1)
The a terms represents a rotational, non-conservative force

The expected entropy production rate is 2a^2 / k
'''

import numpy as np
import freqent.altMethods as alt
from tqdm import tqdm
from drivenBrownianParticle import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# plotting preferences
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

r0 = np.random.randn(2)
nsteps = 1e6
dt = 1e-3
alpha = 2

r = drivenBrownianParticle(dt=dt, r0=r0, nsteps=nsteps)
r.runSimulation(alpha)

# set edges for probability map and velocity field
# assume data does not go beyond ± 5
edges = [np.linspace(-5, 5, 41)] * 2
# get bin centers
dx = np.array([np.diff(e)[0] for e in edges])
xx, yy = np.meshgrid(edges[0][:-1] + dx[0] / 2, edges[1][:-1] + dx[1] / 2)

epr, p, v, edges = alt.spatialEPR(r.pos, dt=dt, bins=edges, verbose=True)

# get theoretical values of probability field and probability flux field
# see supplement in Seara, Machta, Murrell, Nat. Comms. 2021
p_thry = np.exp(-(xx**2 + yy**2) / 2) / (2 * np.pi)
jx = alpha * -yy * p_thry
jy = alpha * xx * p_thry
epr_thry = 2 * alpha**2

print('Theoretical EPR: {s:.3f}'.format(s=epr_thry))
print('Estimated EPR: {s:.3f}'.format(s=epr))

fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.5))

# plot theoretical probability and flux fields
a0 = ax[0, 0].pcolormesh(edges[0][:-1], edges[1][:-1], p_thry,
                         rasterized=True)
q0 = ax[0, 0].quiver(xx, yy, jx, jy,
                     color='w', label=r'$j_\mathrm{thry}$')
plt.quiverkey(q0, X=0.9, Y=0.8, U=jx.mean(),
              label=r'$flux$', labelcolor='w')
cbar0 = fig.colorbar(a0, ax=ax[0, 0])
cbar0.ax.set(title=r'$p$')
ax[0, 0].set(xlim=[-3, 3], ylim=[-3, 3],
             xlabel=r'$x$', ylabel=r'$y$', title='Theory')
ax[0, 0].set_aspect('equal')

# plot estimated probability and flux fields
a1 = ax[0, 1].pcolormesh(edges[0][:-1], edges[1][:-1], p.T,
                         rasterized=True)
q1 = ax[0, 1].quiver(xx, yy, (v[0] * p).T, (v[1] * p).T,
                     color='w', label=r'$j_\mathrm{est}$')
plt.quiverkey(q1, X=0.9, Y=0.8, U=jx.mean(),
              label=r'$flux$', labelcolor='w')
cbar1 = fig.colorbar(a1, ax=ax[0, 1])
cbar1.ax.set(title=r'$p$')
ax[0, 1].set(xlim=[-3, 3], ylim=[-3, 3],
             xlabel=r'$x$', ylabel=r'$y$', title='Estimated')
ax[0, 1].set_aspect('equal')

# plot theoretical and estimated values of probability against each other
ax[1, 0].plot(np.ravel(p_thry), np.ravel(p), 'k.', alpha=0.2)
# plot a line of slope 1 to show how well the estimates match the theory
ax[1, 0].plot([p_thry.min(), p_thry.max()], [p_thry.min(), p_thry.max()],
              'r--', lw=3, label=r'$m = 1$')
ax[1, 0].legend()
ax[1, 0].set(xlim=[-0.01, 0.17], ylim=[-0.01, 0.17],
             xlabel=r'$p_\mathrm{thry}$', ylabel=r'$p_\mathrm{est}$')
ax[1, 0].set_aspect('equal')

# plot theoretical and estimated values of probability fluxes against each other
ax[1, 1].plot(np.ravel(jx), np.ravel((v[0] * p).T), '.',
              alpha=0.2, label=r'$j_x$')
ax[1, 1].plot(np.ravel(jy), np.ravel((v[1] * p).T), '.',
              alpha=0.2, label=r'$j_y$')
# plot a line of slope 1 to show how well the estimates match the theory
ax[1, 1].plot([np.min([jx.min(), jy.min()]), np.max([jx.max(), jy.max()])],
              [np.min([jx.min(), jy.min()]), np.max([jx.max(), jy.max()])],
              'r--', lw=3, label=r'$m = 1$')
ax[1, 1].legend()
ax[1, 1].set(xlim=[-0.25, 0.25], ylim=[-0.25, 0.25],
             xlabel=r'$j_\mathrm{thry}$', ylabel=r'$j_\mathrm{est}$')
ax[1, 1].set_aspect('equal')

plt.tight_layout()
plt.show()
