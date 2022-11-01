"""
Reads in the AbacusSummit 'small' simulation data (cubes of length 500Mpc/h)
and splits the halo positions into quintiles using Enrique's density split
algorithm. For each simulation realization and each density quintile,
saves the x,y,z coordinates for later use.
"""


import os
import numpy as np
from astropy.io import fits
from densitysplit.pipeline import DensitySplit


def get_data_positions(data_fn, split='z', los='z'):
    '''
    Returns halo positions in real or redshift space.
    '''
    with fits.open(data_fn) as hdul:
        mock_data = hdul[1].data
        if split == 'z':
            xgal = mock_data['X_RSD'] if los == 'x' else mock_data['X']
            ygal = mock_data['Y_RSD'] if los == 'y' else mock_data['Y']
            zgal = mock_data['Z_RSD'] if los == 'z' else mock_data['Z']
        else:
            xgal = mock_data['X']
            ygal = mock_data['Y']
            zgal = mock_data['Z']
    return np.c_[xgal, ygal, zgal]


if __name__ == '__main__':
    split = 'r' # working in real space (for now)
    boxsize = 500 # length of simulation cube (comoving Mpc/h)
    smooth_ds = 20 # smoothing radius
    nquantiles = 5 # number of quantiles to use
    los = 'z' # line of sight direction
    cellsize = 5 # size of mesh cells within simulation cube
    
    # need to change path if using redshift/recon space or changing the number
    # of quantiles used (for now, real space with five quantiles)
    save_path = '/home/jgmorawe/results/quantiles/quintiles/real'
    
    # iterates through all the realizations
    sim_nums = np.arange(3000, 5000)
    sim_path_start = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    for sim_num in sim_nums:
        sim_path = os.path.join(
            sim_path_start,
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        # first checks if simulation exists (some fail), if not skips
        if os.path.exists(sim_path):
            data_positions = get_data_positions(
                data_fn=sim_path, split=split, los=los)
            densitysplit = DensitySplit(
                data_positions=data_positions, boxsize=boxsize)
            density = densitysplit.get_density(
                smooth_radius=smooth_ds, cellsize=cellsize, 
                sampling_positions=data_positions)
            quantiles = densitysplit.get_quantiles(nquantiles=nquantiles)
            # saves results
            for i in range(nquantiles):
                np.save(os.path.join(
                    save_path, 'sim{0}_ds{1}.npy'.format(sim_num, i+1)),
                    quantiles[i])
        else:
            print('Simulation {} failed!'.format(sim_num))