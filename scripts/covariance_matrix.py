"""
This script uses multiple realizations of the same cosmology from the 
AbacusSummit data, runs the usual density split and power spectrum data,
to compute the covariance matrix (begins by generating the power spectra
for auto/cross for each quintile and real/redshift space, for all of the 
different realizations, and then iterates over the saved results to generate 
the covariance matrix).
"""


import os
import time
import pickle
import numpy as np
from astropy.io import fits
from densitysplit.pipeline import DensitySplit
from pypower import CatalogFFTPower


sim_path = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
power_path = '/home/jgmorawe/results/power_spectra'
quintile_path = '/home/jgmorawe/results/quintiles'


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
    
    
    boxsize = 500
    smooth_ds = 20
    n_quantiles = 5
    los = 'z'
    cellsize = 5
    
    t0 = time.time()
    
    for sim_num in range(3000, 5000):
        folder_path = os.path.join(sim_path,
            'AbacusSummit_small_c000_ph{}'.format(sim_num))
        # checks if there is simulation data for particular simulation number
        file_path = os.path.join(folder_path, 'halos/z0.575',
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        if os.path.exists(file_path):
            for split in ['z', 'r']:
                data_positions = get_data_positions(file_path, split=split, los=los)
                ndata = len(data_positions)
                ds = DensitySplit(data_positions, boxsize)
                sampling_positions = np.random.uniform(0, boxsize, (5 * ndata, 3))
                density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                                         sampling_positions=sampling_positions)
                quantiles = ds.get_quantiles(nquantiles=5)
                rand_positions1 = quantiles[0]
                rand_positions2 = quantiles[1]
                rand_positions3 = quantiles[2]
                rand_positions4 = quantiles[3]
                rand_positions5 = quantiles[4]
                rand_position_list = [rand_positions1, rand_positions2,
                                      rand_positions3, rand_positions4,
                                      rand_positions5]
                density_labels = ['1', '2', '3', '4', '5']
                for i in range(len(rand_position_list)):
                    rand_positions = rand_position_list[i]
                    density_label = density_labels[i]
                    # auto power first, then cross power second
                    auto_result = CatalogFFTPower(data_positions1=rand_positions,
                                             edges=np.linspace(0, 0.65, 66), 
                                             boxsize=boxsize, cellsize=cellsize,
                                             los=los, position_type='pos')
                    cross_result = CatalogFFTPower(data_positions1=rand_positions,
                                                   data_positions2=data_positions,
                                                   edges=np.linspace(0, 0.65, 66),
                                                   boxsize=boxsize, cellsize=cellsize,
                                                   los=los, position_type='pos')
                    # pickles the results to files
                    pickle.dump(auto_result, open(
                        os.path.join(power_path, 'Auto_{0}_{1}_{2}.pkl'.format(split, density_label, sim_num)) , 'wb'))
                    pickle.dump(cross_result, open(
                        os.path.join(power_path, 'Cross_{0}_{1}_{2}.pkl'.format(split, density_label, sim_num)), 'wb'))
                
        else: # skips if simulation failed and thus no path exists
            continue
    
    print('Power spectra generated in: ', (time.time()-t0) // 60, 'minutes.')
    
    
    
    # Now reads in the power spectra to generate the covariance matrix
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        