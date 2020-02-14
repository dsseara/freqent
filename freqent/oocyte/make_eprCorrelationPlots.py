import numpy as np
import matplotlib.pyplot as plt

epr = np.array([0.047277,
                0.032366,
                0.173335,
                0.122434,
                0.110645,
                0.030853,
                0.054728,
                0.032132,
                0.027112,
                0.043444,
                0.039702,
                # 1.209387,
                0.043289,
                0.020511,
                0.037424,
                0.025866,
                0.050951])



tau = np.array([3.63E+01,
                5.69E+01,
                3.47E+02,
                2.47E+02,
                5.02E+02,
                1.98E+02,
                2.81E+02,
                1.51e+02,
                1.28E+02,
                2.77E+02,
                7.65E+01,
                # 2.00E+02,
                2.39E+02,
                0,
                1.81E+02,
                2.07E+02,
                288.26950131])

alpha = np.array([0.99,
                  0.98,
                  1.48,
                  1.52,
                  1.45,
                  1.33,
                  1.3,
                  1.23,
                  0.99,
                  1.33,
                  1.05,
                  # 1.06,
                  1.09,
                  0.98,
                  1.03,
                  0.98,
                  1.08])

files = ['140706_08',
         '140706_09',
         '140713_08',
         '140713_09',
         '140717_01',
         '140717_09',
         '140717_13',
         '140817_05',
         '160403_09',
         '160403_12',
         '160403_14',
         '160915_09',
         '160915_12',
         '161001_04',
         '161025_01',
         '171230_04',
         '171230_06']

fig, ax = plt.subplots(1, 3)
# ax[0].plot(epr, tau, 'ko')
ax[0].plot(epr[np.where(alpha < 1.1)[0]], tau[np.where(alpha < 1.1)[0]], 'bo')
ax[0].plot(epr[np.where(np.logical_and(alpha > 1.2, alpha < 1.4))[0]],
           tau[np.where(np.logical_and(alpha > 1.2, alpha < 1.4))[0]],
           'ro')
ax[0].plot(epr[np.where(alpha > 1.4)[0]], tau[np.where(alpha > 1.4)[0]], 'ko')
ax[0].set(xlabel='epr', ylabel='tau')
# ax[1].plot(alpha, tau, 'ko')
ax[1].plot(alpha[np.where(alpha < 1.1)[0]], tau[np.where(alpha < 1.1)[0]], 'bo')
ax[1].plot(alpha[np.where(np.logical_and(alpha > 1.2, alpha < 1.4))[0]],
           tau[np.where(np.logical_and(alpha > 1.2, alpha < 1.4))[0]],
           'ro')
ax[1].plot(alpha[np.where(alpha > 1.4)[0]], tau[np.where(alpha > 1.4)[0]], 'ko')
ax[1].set(xlabel='alpha', ylabel='tau')
# ax[2].plot(epr, alpha, 'ko')
ax[2].plot(epr[np.where(alpha < 1.1)[0]], alpha[np.where(alpha < 1.1)[0]], 'bo')
ax[2].plot(epr[np.where(np.logical_and(alpha > 1.2, alpha < 1.4))[0]],
           alpha[np.where(np.logical_and(alpha > 1.2, alpha < 1.4))[0]],
           'ro')
ax[2].plot(epr[np.where(alpha > 1.4)[0]], alpha[np.where(alpha > 1.4)[0]], 'ko')
ax[2].set(xlabel='epr', ylabel='alpha')
