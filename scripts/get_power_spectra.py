"""
Generates cross power spectra for the density quantiles (the positions are 
generated randomly, and then partitioned into quantiles according to the 
underlying smoothed density field of the halo positions).
"""


import os
import time
import pickle
import numpy as np
from astropy.io import fits
from densitysplit.pipeline import DensitySplit
from pypower import CatalogFFTPower



def get_data_positions(data_fn, split='z', los='z'):
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
    
    t0 = time.time()
    
    sim_folder_path = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    power_folder_path = '/home/jgmorawe/results/power_spectra/real'
    
    boxsize = 500 # size of simulation cube
    smooth_ds = 20 # smoothing radius
    cellsize = 5 # cell size within the simulation cube
    los = 'z' # line of sight direction
    split = 'r' # real space (deal with redshift/reconstruction space later)
    
    # iterates through each simulation realization (some have errors)
    for sim_num in np.arange(3000, 5000):
        
        sim_path = os.path.join(
            sim_folder_path, 
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        
        if os.path.exists(sim_path):
            
            halo_positions = get_data_positions(
                data_fn=sim_path, split='r', los='z')
            ndata = len(halo_positions)
            ds_object = DensitySplit(
                data_positions=halo_positions, boxsize=boxsize)
            # generates random positions to fill up the sample volume
            random_positions = np.random.uniform(0, boxsize, (5*ndata, 3))
            density = ds_object.get_density(
                smooth_radius=smooth_ds, cellsize=cellsize,
                sampling_positions=random_positions)
            quantiles = ds_object.get_quantiles(nquantiles=5)
            ds1_positions = quantiles[0]
            ds5_positions = quantiles[-1]
            
            cross1 = CatalogFFTPower(
                data_positions1=ds1_positions, data_positions2=halo_positions,
                edges=np.arange(2*np.pi/boxsize, 2*np.pi/(smooth_ds/2),
                                2*np.pi/boxsize),
                ells=(0), los=los, boxsize=boxsize, cellsize=cellsize,
                position_type='pos')
            
            cross5 = CatalogFFTPower(
                data_positions1=ds5_positions, data_positions2=halo_positions,
                edges=np.arange(2*np.pi/boxsize, 2*np.pi/(smooth_ds/2),
                                2*np.pi/boxsize),
                ells=(0), los=los, boxsize=boxsize, cellsize=cellsize,
                position_type='pos')
            
            pickle.dump(
                cross1,
                open(os.path.join(power_folder_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross5,
                open(os.path.join(power_folder_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num)), 'wb'))
            
        else:
            print('Simulation {} failed.'.format(sim_num))


    print('Script executed in {} seconds.'.format(round(time.time()-t0, 1)))