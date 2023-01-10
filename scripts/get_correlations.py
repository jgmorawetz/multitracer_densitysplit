"""
Generates auto and cross correlation functions for the DS quantiles.
"""



import os
import time
import pickle
import numpy as np
from astropy.io import fits
from pycorr import TwoPointCorrelationFunction



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
    
    t0 = time.time()
    
    boxsize = 500
    smooth_ds = 20
    los = 'z'
    split = 'r'
    
    sim_folder_path = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    quantile_path = '/home/jgmorawe/results/quantiles/quintiles/real'
    corr_folder_path = '/home/jgmorawe/results/correlation/monopole/real/ds'
    
    for sim_num in range(3000, 5000):
        
        sim_path = os.path.join(
            sim_folder_path,
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        
        if os.path.exists(sim_path):

            halo_positions = get_data_positions(
                data_fn=sim_path, split=split, los=los)
            ds1_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds1.npy'.format(sim_num)))
            ds5_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds5.npy'.format(sim_num)))
            
            edges = np.linspace(10, 150, 29)
            
            # cross correlation of quintiles with halo positions
            cross_ds1_halo = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=ds1_positions,
                data_positions2=halo_positions, position_type='pos', los=los,
                boxsize=boxsize)
            cross_ds5_halo = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=ds5_positions,
                data_positions2=halo_positions, position_type='pos', los=los,
                boxsize=boxsize)
            
            # cross correlation of quintiles with each other
            cross_ds1_ds5 = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=ds1_positions,
                data_positions2=ds5_positions, position_type='pos', los=los,
                boxsize=boxsize)
            
            # auto correlation of halo positions
            auto_halo_halo = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=halo_positions,
                position_type='pos', los=los, boxsize=boxsize)
            
            # auto correlation of quintiles with themselves
            auto_ds1_ds1 = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=ds1_positions,
                position_type='pos', los=los, boxsize=boxsize)
            auto_ds5_ds5 = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=ds5_positions,
                position_type='pos', los=los, boxsize=boxsize)

            # saves to file for later usage
            pickle.dump(
                cross_ds1_halo, open(os.path.join(corr_folder_path,
                'sim{}_cross_ds1_halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross_ds5_halo, open(os.path.join(corr_folder_path,
                'sim{}_cross_ds5_halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross_ds1_ds5, open(os.path.join(corr_folder_path,
                'sim{}_cross_ds1_ds5.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto_halo_halo, open(os.path.join(corr_folder_path, 
                'sim{}_auto_halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto_ds1_ds1, open(os.path.join(corr_folder_path,
                'sim{}_auto_ds1_ds1.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto_ds5_ds5, open(os.path.join(corr_folder_path,
                'sim{}_auto_ds5_ds5.pkl'.format(sim_num)), 'wb'))
            
            print('Simulation {} done.'.format(sim_num))

        else:
            print('Simulation {} failed.'.format(sim_num))
    
    print('Script executed in {} seconds.'.format(time.time()-t0))