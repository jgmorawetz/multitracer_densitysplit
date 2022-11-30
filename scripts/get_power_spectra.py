"""
Generates auto and cross power spectra for the DS quantiles.
"""



import os
import time
import pickle
import numpy as np
from astropy.io import fits
from pypower import CatalogFFTPower, mpi



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
    
    # multiprocessing
    mpicomm = mpi.COMM_WORLD
    mpiroot = 0
    
    # set the relevant parameters
    boxsize = 500
    smooth_ds = 20
    cellsize = 5
    los = 'z'
    split = 'r'
    
    # relevant file paths
    sim_folder_path = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    quantile_path = '/home/jgmorawe/results/quantiles/quintiles/real'
    power_folder_path = '/home/jgmorawe/results/power_spectra/monopole/real'
    
    # first retrieves all the simulation numbers without errors
    valid_sim_nums = []
    for sim_num in range(3000, 5000):
        sim_path = os.path.join(
            sim_folder_path,
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        if os.path.exists(sim_path):
            valid_sim_nums.append(sim_num)
            
    # sets bin edges for power
    edges = np.linspace(0.01, 0.5, 451)
    
    # generates auto power spectra first
    for sim_num in valid_sim_nums:
        for quintile in ['1', '5']:
            if mpicomm.rank == mpiroot:
                ds_positions = np.load(
                    os.path.join(quantile_path, 'sim{0}_ds{1}.npy'.format(sim_num, quintile)))
            else:
                ds_positions = None
            auto = CatalogFFTPower(
                data_positions1=ds_positions, edges=edges, ells=(0), los=los,
                boxsize=boxsize, cellsize=cellsize, position_type='pos',
                mpicomm=mpicomm, mpiroot=mpiroot)
            pickle.dump(
                auto, open(os.path.join(power_folder_path, 
                'sim{0}_auto_ds{1}.pkl'.format(sim_num, quintile)), 'wb'))
    print('Auto Power Spectra completed!')
    
    # generates cross power spectra with the halo positions next
    for sim_num in valid_sim_nums:
        for quintile in ['1', '5']:
            if mpicomm.rank == mpiroot:
                sim_path = os.path.join(
                    sim_folder_path,
                    'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
                    'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
                ds_positions = np.load(
                    os.path.join(quantile_path, 'sim{0}_ds{1}.npy'.format(sim_num, quintile)))
                halo_positions = get_data_positions(
                    data_fn=sim_path, split=split, los=los)
            else:
                ds_positions = None
                halo_positions = None
            cross = CatalogFFTPower(
                data_positions1=ds_positions, data_positions2=halo_positions,
                edges=edges, ells=(0), los=los,
                boxsize=boxsize, cellsize=cellsize, position_type='pos',
                mpicomm=mpicomm, mpiroot=mpiroot)
            pickle.dump(
                cross, open(os.path.join(power_folder_path, 
                'sim{0}_cross_ds{1}-halo.pkl'.format(sim_num, quintile)), 'wb'))
    print('Cross-Halo Power Spectra completed!')
    
    # generates cross power spectra with each other finally
    for sim_num in valid_sim_nums:
        if mpicomm.rank == mpiroot:
            ds1_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds1.npy'.format(sim_num)))
            ds5_positions = np.load(
                os.path.join(quantile_path, 'sim{}_ds5.npy'.format(sim_num)))
        else:
            ds1_positions = None
            ds5_positions = None
        cross = CatalogFFTPower(
            data_positions1=ds1_positions, data_positions2=ds5_positions,
            edges=edges, ells=(0), los=los,
            boxsize=boxsize, cellsize=cellsize, position_type='pos',
            mpicomm=mpicomm, mpiroot=mpiroot)
        pickle.dump(
            cross, open(os.path.join(power_folder_path, 
            'sim{}_cross_ds1-ds5.pkl'.format(sim_num), 'wb')))
    print('Cross-Halo Power Spectra completed!')
    
    
    
    
    
    print('Script executed in {} seconds.'.format(time.time()-t0))