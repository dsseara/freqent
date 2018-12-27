'''
Non-dimensionalized Brusselator model:

dx/dt = a - (1 + b) * x + x^2 * y
dy/dt = b * x - x^2 * y

a,b > 0 are parameters
x,y > 0 are non-dimensionalized concentrations

Fixed point at (a, b/a)
Critical point at bc = 1 + a^2
attractor if b < bc (orbits fall into fixed point)
repeller if b > bc (enters limit cycle)
'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.close('all')

t = np.linspace(0, 100, 2001)
a = 1
bc = 1 + a**2


def brusselator(r, t, a, b):
    '''
    r = [x, y]
    '''
    x, y = r
    drdt = [a - (1 + b) * x + x**2 * y, b * x - x**2 * y]
    return drdt


bArray = [4]
# cmap = mpl.cm.get_cmap('tab10')
# normalize = mpl.colors.Normalize(vmin=min(bArray), vmax=max(bArray))
# colors = [cmap(normalize(value)) for value in bArray]

fig, ax = plt.subplots()

for bInd, b in enumerate(bArray):
    r0 = np.random.rand(2) * bc
    sol = odeint(brusselator, r0, t, args=(a, b))
    ax.plot(sol[:, 0], sol[:, 1], color='C{0}'.format(bInd))
    ax.plot(a, b / a, marker='X', markersize=10, markeredgecolor='k', color='C{0}'.format(bInd))

# Optionally add a colorbar
# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
# cbar.ax.set_title(r'$b$')

# legend_elements = [mpl.lines.Line2D([0], [0], markersize=10, marker='X',
#                                     linestyle='None', markeredgecolor='k',
#                                     color='k', label='fixed point'),
#                    mpl.lines.Line2D([0], [0], color='C{0}'.format(ind)) for ind in len(bArray)]

# ax.legend(handles=legend_elements, loc='best')
ax.set(xlabel='x', ylabel='y', title=r'Brusselator, $a$={a}, $b_c$={bc}'.format(a=a, bc=bc))

plt.show()
