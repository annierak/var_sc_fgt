from __future__ import division

import math
import numpy as np
import time



def _puff_conc_dist(x, y, z, px, py, pz, r_sq):
    # calculate Gaussian puff concentration distribution
    _ampl_const = 1.
    return (
        _ampl_const / r_sq**1.5 *\
        np.exp(-((x - px)**2 + (y - py)**2 + (z - pz)**2) / (2 * r_sq))    )

def _puff_conc_dist_for_loop(x, y, px, py, pz, r_sq):
    # calculate Gaussian puff concentration distribution
    _ampl_const = 1.
    z = np.zeros(np.shape(x))
    px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
    to_fill = np.zeros((len(px),len(px)))
    for i in range(len(px)):
        for j in range(len(x)):
            t = time.time()
            to_fill[i,j] = _ampl_const / r_sq[i]**1.5 *\
            np.exp(-((x[j] - px[i])**2 + (y[j] - py[i])**2 + (z[j] - pz[i])**2) / (2 * r_sq[i]))
            print(time.time()-t)

    return to_fill
def calc_conc_list(puff_array, x, y, z=0):
    """
    puff_array : numpy-array-like of Puff objects
        Collection of currently alive puff instances at a particular
        time step which it is desired to calculate concentration field
        values from.
    x : (np) numpy-array-like of floats
        1D array of x-coordinates of points.
    y : (np) numpy-array-like of floats
        1D array of y-coordinates of points.
    z : float
        z-coordinate (height) of plane.
    """
    # filter for non-nan puff entries and separate properties for
    # convenience
    px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
    na = np.newaxis
    # sq_dist= (x - px)**2 + (y - py)**2 + (z - pz)**2

    return _puff_conc_dist(x[:, na], y[:, na], z, px[na, :],
                                py[na, :], pz[na, :], r_sq[na, :]).sum(-1)

num_puffs = 50000
num_flies = 2000

arena_size = 10

px,py,pz,r_sq = np.random.uniform(-arena_size,arena_size,num_puffs),\
    np.random.uniform(-arena_size,arena_size,num_puffs),\
    np.random.uniform(-arena_size,arena_size,num_puffs),\
    np.random.uniform(0,1,num_puffs)

puff_array = np.array([px,py,pz,r_sq]).T

x,y = np.random.uniform(-arena_size,arena_size,num_flies),\
    np.random.uniform(-arena_size,arena_size,num_flies)

t1 = time.time()
conc_vals = calc_conc_list(puff_array,x,y)
print(np.shape(conc_vals))
print(time.time()-t1)

t1 = time.time()
conc_vals = _puff_conc_dist_for_loop(x,y,px, py, pz, r_sq)
print(time.time()-t1)
