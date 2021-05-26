'''
Simulation of a Brownian particle in 2D in a harmonic trap
and acted upon by a rotational force
'''

import numpy as np
from tqdm import tqdm


class drivenBrownianParticle():
    '''
    class for solving the N-D overdamped Langevin equation. Models a harmonic
    potential plus a rotational force in the first two dimensions
    gamma * dr = F * dt + xi * dt
    gamma is a drag coefficient
           ---     ---
           | -k   -a |
    F(r) = |  a   -k |
           ---     ---
        k is spring constant, along diagonals
        a is strength of rotational force,
        xi is zero mean Gaussian white noise with
        <xi> = 0
        <xi_i (t) xi_j (t')> = 2*D*gamma^2 * delta_ij * delta(t - t')
            D is diffusion constant, D = kB*T/gamma

    For simulations, we non-dimensionalize the equations of motion with
    time scale gamma/k and length scale sqrt(D * gamma / k).
    a1 and a2 denote the strength of coupling first and second dimension with
    the other, respectively.
    '''

    def __init__(self, dt, r0, nsteps=1e6):

        self.dt = dt  # time step in simulation time
        self.nsteps = int(nsteps)  # total number of steps
        self.r0 = r0
        self.ndim = len(r0)  # number of dimensions in simulation
        if self.ndim < 2:
            raise ValueError('Need at least 2 dimensions, currently have '
                             '{0}'.format(self.ndim))

        # data
        self.t = np.linspace(0, self.nsteps * self.dt, self.nsteps + 1)
        self.pos = np.zeros((self.ndim, self.nsteps + 1))
        self.pos[:, 0] = r0

    def reset(self):
        self.__init__(self.dt, self.r0, self.nsteps)

    def deterministicForce(self, r, alpha):
        spring = np.diag(-1 * np.ones(self.ndim))  # spring force along diagonal
        rotate = np.zeros((self.ndim, self.ndim))  # construct rotational force matrix
        rotate[(1, 0), (0, 1)] = (alpha, -alpha)  # fill in appropriate elements

        return np.matmul(spring + rotate, r)

    def noise(self):
        '''
        Gaussian white noise
        '''
        return np.sqrt(2 / self.dt) * np.random.randn(self.ndim)

    def runSimulation(self, alpha):
        # self.reset()

        for index, time in enumerate(tqdm(self.t[1:])):
            pos_old = self.pos[:, index]
            pos_new = pos_old + (self.deterministicForce(pos_old, alpha) + self.noise()) * self.dt
            self.pos[:, index + 1] = pos_new
