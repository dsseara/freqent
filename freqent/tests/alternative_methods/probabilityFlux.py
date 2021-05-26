import numpy as np
from tqdm import tqdm
import pandas as pd
import pdb


def probabilityFlux(data, dt=1, bins=10):
    '''
    Compute the flux field of a time series data
    *** ONLY WORKS FOR 2D DATA RIGHT NOW ***

    Parameters
    ----------
    data : array
        DxN array of time series data with N time points in D
        dimensions
    dt : float, optional
        size of time step between data points. Defaults to 1
    bins : sequence or int, optional
        The bin specification:
            - A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            - The number of bins for each dimension (nx, ny, … =bins)
            - The number of bins for all dimensions (nx=ny=…=bins).
        Defaults to 10

    Results
    -------
    prob_map : array
        histogram of probabilities for each state
    velocity_field : array
        vector field of velocities in shape D x [nbins]
    edges : array
        edges used to discretize space

    See also
    --------

    Example
    -------

    '''

    data = np.asarray(data)
    ndim, nt = data.shape
    T = nt * dt

    if ndim != 2:
        raise ValueError('This function only works for 2D data')

    prob_map, edges = np.histogramdd(data, bins=bins, density=True)
    nbins = [len(e) - 1 for e in edges]

    # calculate velocities
    velocity = np.gradient(data, dt, axis=0)

    # use digitize over each dimension to find list of states
    states = np.array([np.digitize(d, e) - 1 for d, e in zip(data, edges)])

    # preallocate velocity field
    velocity_field = np.zeros((ndim, *nbins))

    # use pandas groupby to perform average in every bin
    velocity_df = pd.DataFrame({'x': data[0],
                                'y': data[1],
                                'vx': velocity[0],
                                'vy': velocity[1],
                                'xstate': states[0],
                                'ystate': states[1]})

    velocity_avg = velocity_df.groupby(['xstate', 'ystate']).mean()[['vx', 'vy']].reset_index()

    # assign averaged velocities to array elements
    for x, y, vx, vy in zip(velocity_avg.xstate, velocity_avg.ystate, velocity_avg.vx, velocity_avg.vy):
        velocity_field[0, x, y] = vx
        velocity_field[1, x, y] = vy

    return prob_map, velocity_field, edges
