"""
Reads in the already generated quantile data from the AbacusSummit 'small'
simulation realizations, and produces auto and cross power spectrums for the 
various density quantiles.
"""


import os
import time
import pickle
import numpy as np
from pypower import CatalogFFTPower


if __name__ == '__main__':
    
    t0 = time.time()
    
    # for now, only generates power spectra for real space
    quantile_path = '/home/jgmorawe/results/quantiles/quintiles/real'
    save_path = '/home/jgmorawe/results/power_spectra/real'
    sim_nums = list(map(lambda x: int(x.split('sim')[-1].split('_')[0]),
                        os.listdir(quantile_path)))
    boxsize = 500
    cellsize = 5
    los = 'z'
    
    # assumes we are dealing only with case of quintiles for now;
    # also, we have already filtered out bad simulation numbers
    for sim_num in sim_nums:
        ds_groups = []
        for i in range(1, 6):
            ds_groups.append(np.load(os.path.join(
                quantile_path, 'sim{0}_ds{1}.npy'.format(sim_num, i))))
        # concatenates all quintiles to get the full halo field
        ds_all = np.vstack(ds_groups)
        # for now, just uses ds1 and ds5 for the power spectra, since those
        # appear to be the best tracers to use
        ds1_positions = ds_groups[0]
        ds5_positions = ds_groups[-1]
        ds_positions = ds_all
        
        # takes auto power spectrum (each quantile with itself) and then the
        # cross power spectrum (of each quintile with the full halo field);
        # for the power spectrum sets the wavenumber limits and bin width such
        # that the boxsize and cellsize and upper lower bounds
        auto1 = CatalogFFTPower(
            data_positions1=ds1_positions, 
            edges=np.arange(0, 2*np.pi/cellsize, 2*np.pi/boxsize),
            # only uses the monopole for real space (may need to change
            # for redshift space)
            ells=(0), los=los, boxsize=boxsize, cellsize=cellsize, 
            position_type='pos')
        auto5 = CatalogFFTPower(
            data_positions1=ds5_positions, 
            edges=np.arange(0, 2*np.pi/cellsize, 2*np.pi/boxsize),
            # only uses the monopole for real space (may need to change
            # for redshift space)
            ells=(0), los=los, boxsize=boxsize, cellsize=cellsize, 
            position_type='pos')
        cross1 = CatalogFFTPower(
            data_positions1=ds1_positions, data_positions2=ds_positions,
            edges=np.arange(0, 2*np.pi/cellsize, 2*np.pi/boxsize),
            ells=(0), los=los, boxsize=boxsize, cellsize=cellsize,
            position_type='pos')
        cross5 = CatalogFFTPower(
            data_positions1=ds5_positions, data_positions2=ds_positions,
            edges=np.arange(0, 2*np.pi/cellsize, 2*np.pi/boxsize),
            ells=(0), los=los, boxsize=boxsize, cellsize=cellsize,
            position_type='pos')
        # pickles each result to file, to retrieve results later
        pickle.dump(
            auto1, open(os.path.join(
            save_path, 'sim{}_monopole_auto_ds1.pkl'.format(sim_num)), 'wb'))
        pickle.dump(
            auto5, open(os.path.join(
            save_path, 'sim{}_monopole_auto_ds5.pkl'.format(sim_num)), 'wb'))
        pickle.dump(
            cross1, open(os.path.join(
            save_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num)), 'wb'))
        pickle.dump(
            cross5, open(os.path.join(
            save_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num)), 'wb'))
    
    print('Script executed in {} seconds.'.format(round(time.time()-t0, 1)))