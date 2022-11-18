"""
Generates random positions within each quantile.
"""



import os
import numpy as np; np.random.seed(0)
from astropy.io import fits
from densitysplit.pipeline import DensitySplit



def get_data_positions(data_fn, split='z', los='z'):
    """
    Retrieves halo positions in real or redshift space.
    """
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

    # set the relevant parameters
    boxsize = 500
    smooth_ds = 20
    cellsize = 5
    nquantiles = 5
    los = 'z'
    split = 'r'
    
    save_path = '/home/jgmorawe/results/quantiles/quintiles/real'
    sim_nums = np.arange(3000, 5000)
    sim_path_start = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    
    for sim_num in sim_nums:
        
        sim_path = os.path.join(
            sim_path_start,
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        
        if os.path.exists(sim_path):
            halo_positions = get_data_positions(
                data_fn=sim_path, split=split, los=los)
            ndata = len(halo_positions)
            ds_object = DensitySplit(
                data_positions=halo_positions, boxsize=boxsize)
            random_positions = np.random.uniform(0, boxsize, (5*ndata, 3))
            density = ds_object.get_density(
                smooth_radius=smooth_ds, cellsize=cellsize,
                sampling_positions=random_positions)
            quantiles = ds_object.get_quantiles(nquantiles=nquantiles)
            ds1_positions = quantiles[0]
            ds2_positions = quantiles[1]
            ds3_positions = quantiles[2]
            ds4_positions = quantiles[3]
            ds5_positions = quantiles[4]
            np.save(os.path.join(
                save_path, 'sim{}_ds1.npy'.format(sim_num)), ds1_positions)
            np.save(os.path.join(
                save_path, 'sim{}_ds2.npy'.format(sim_num)), ds2_positions)
            np.save(os.path.join(
                save_path, 'sim{}_ds3.npy'.format(sim_num)), ds3_positions)
            np.save(os.path.join(
                save_path, 'sim{}_ds4.npy'.format(sim_num)), ds4_positions)
            np.save(os.path.join(
                save_path, 'sim{}_ds5.npy'.format(sim_num)), ds5_positions)

        else:
            print('Simulation {} failed.'.format(sim_num))