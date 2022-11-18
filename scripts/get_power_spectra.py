"""
Generates auto and cross power spectra for the DS quantiles.
"""



import os
import time
import pickle
import numpy as np
from astropy.io import fits
from pypower import CatalogFFTPower



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
    
    # set the relevant parameters
    boxsize = 500
    smooth_ds = 20
    cellsize = 5
    los = 'z'
    split = 'r'
    
    sim_folder_path = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    quantile_path = '/home/jgmorawe/results/quantiles/quintiles/real'
    power_folder_path = '/home/jgmorawe/results/power_spectra/monopole/real'
    
    for sim_num in range(3000, 5000):
        
        sim_path = os.path.join(
            sim_folder_path,
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        
        if os.path.exists(sim_path):
            # halo and quintile positions
            halo_positions = get_data_positions(
                data_fn=sim_path, split=split, los=los)
            ds1_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds1.npy'.format(sim_num)))
            ds2_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds2.npy'.format(sim_num)))
            ds3_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds3.npy'.format(sim_num)))
            ds4_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds4.npy'.format(sim_num)))
            ds5_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds5.npy'.format(sim_num)))
            
            edges = np.arange(2*np.pi/boxsize, 
                              2*np.pi/(boxsize/10)+2*np.pi/boxsize,
                              2*np.pi/boxsize)
            
            # power spectrum of each quintile with halo positions
            cross1 = CatalogFFTPower(
                data_positions1=ds1_positions, data_positions2=halo_positions,
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            cross2 = CatalogFFTPower(
                data_positions1=ds2_positions, data_positions2=halo_positions,
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            cross3 = CatalogFFTPower(
                data_positions1=ds3_positions, data_positions2=halo_positions,
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            cross4 = CatalogFFTPower(
                data_positions1=ds4_positions, data_positions2=halo_positions,
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            cross5 = CatalogFFTPower(
                data_positions1=ds5_positions, data_positions2=halo_positions,
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            
            # auto spectrum of each quintile with itself (real space)
            auto1 = CatalogFFTPower(
                data_positions1=ds1_positions, 
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            auto2 = CatalogFFTPower(
                data_positions1=ds2_positions, 
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            auto3 = CatalogFFTPower(
                data_positions1=ds3_positions, 
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            auto4 = CatalogFFTPower(
                data_positions1=ds4_positions, 
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            auto5 = CatalogFFTPower(
                data_positions1=ds5_positions, 
                edges=edges, ells=(0), los=los, boxsize=boxsize, 
                cellsize=cellsize, position_type='pos')
            
            # saves to file for later usage
            pickle.dump(
                cross1, open(os.path.join(power_folder_path, 
                'sim{}_cross_ds1halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross2, open(os.path.join(power_folder_path, 
                'sim{}_cross_ds2halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross3, open(os.path.join(power_folder_path, 
                'sim{}_cross_ds3halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross4, open(os.path.join(power_folder_path, 
                'sim{}_cross_ds4halo.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                cross5, open(os.path.join(power_folder_path, 
                'sim{}_cross_ds5halo.pkl'.format(sim_num)), 'wb'))
            
            pickle.dump(
                auto1, open(os.path.join(power_folder_path, 
                'sim{}_auto_ds1.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto2, open(os.path.join(power_folder_path, 
                'sim{}_auto_ds2.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto3, open(os.path.join(power_folder_path, 
                'sim{}_auto_ds3.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto4, open(os.path.join(power_folder_path, 
                'sim{}_auto_ds4.pkl'.format(sim_num)), 'wb'))
            pickle.dump(
                auto5, open(os.path.join(power_folder_path, 
                'sim{}_auto_ds5.pkl'.format(sim_num)), 'wb'))
            
        else:
            print('Simulation {} failed.'.format(sim_num))
    
    print('Script executed in {} seconds.'.format(time.time()-t0))