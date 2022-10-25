"""
This script is used to read in AbacusSummit simulation data (a cube of
x,y,z positions in redshift and real space), and split the positions into
density quantiles.
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
    split = 'r' # work in real space for now
    boxsize = 500 # length of simulation cube
    smooth_ds = 20 # smooth radius
    nquantiles = 5 # number of percentiles to use
    los = 'z' # line of sight direction
    cellsize = 5 # size of the cells within the cube
    
    output_folder = '/home/jgmorawe/results/quintiles/real'
    
    sim_nums = np.arange(3000, 5000)
    sim_path_start = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    
    for sim_num in sim_nums:
        sim_path = os.path.join(sim_path_start, 
                                'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
                                'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        if os.path.exists(sim_path):
            data_positions = get_data_positions(sim_path, 'r', 'z')
            ds = DensitySplit(data_positions, boxsize)
            density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                                     sampling_positions=data_positions)
            quantiles = ds.get_quantiles(nquantiles=5)
            # saves results
            np.save(os.path.join(output_folder, 'sim_{}_ds1.npy'.format(sim_num)), quantiles[0])
            np.save(os.path.join(output_folder, 'sim_{}_ds2.npy'.format(sim_num)), quantiles[1])
            np.save(os.path.join(output_folder, 'sim_{}_ds3.npy'.format(sim_num)), quantiles[2])
            np.save(os.path.join(output_folder, 'sim_{}_ds4.npy'.format(sim_num)), quantiles[3])
            np.save(os.path.join(output_folder, 'sim_{}_ds5.npy'.format(sim_num)), quantiles[4])
        else:
            print('No data for {}'.format(sim_num))