"""
This script is used to generate power spectra for real and redshift space
positions for the AbacusSummit data.
"""


import os
import numpy as np
import pickle
from pypower import CatalogFFTPower
#from pypower import setup_logging
from astropy.io import fits


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
    # repeats procedure for both 'z' (redshift space) and 'r' (real space)
    for split in ['z', 'r']:
        # reads in the quintile data positions
        quintile_path = os.path.join(
            '/home/jgmorawe/results/quintiles/',
            'quantiles_{}split_base_c000_ph000_z0.575_nden3.2e-04.npy'.format(
                split))
        quintile_data = np.load(quintile_path)
        density_1_positions = quintile_data[0]
        density_2_positions = quintile_data[1]
        density_3_positions = quintile_data[2]
        density_4_positions = quintile_data[3]
        density_5_positions = quintile_data[4]
        del(quintile_data)
        # reads in the full redshift space halo positions
        data_dir = os.path.join('/home/jgmorawe/projects/rrg-wperciva/',
                                'AbacusSummit/AbacusSummit_base_c000_ph000/',
                                'halos/z0.575')
        # uses cosmo = 0, phase = 0, redshift = 0.575 (may need to adjust)
        data_fn = os.path.join(data_dir,
                    'halos_base_c000_ph000_z0.575_nden3.2e-04.fits')
        all_positions = get_data_positions(data_fn=data_fn, split='z', los='z')
        density_position_list = [density_1_positions, density_2_positions,
                                 density_3_positions, density_4_positions,
                                 density_5_positions]
        del(density_1_positions, density_2_positions, density_3_positions,
            density_4_positions, density_5_positions)
        density_labels = ['1', '2', '3', '4', '5']
        save_path = '/home/jgmorawe/results/power_spectra'
        
        # first computes the auto power spectrum for each quintile's positions
        for i in range(len(density_position_list)):
            density_positions = density_position_list[i]
            density_label = density_labels[i]
           # setup_logging()
            result = CatalogFFTPower(
                data_positions1=density_positions, 
                edges=np.linspace(0, 1, 151),
                boxsize=2000, cellsize=5, los='z', position_type='pos')
            pickle.dump(result, open(os.path.join(save_path, 'Auto_{0}_{1}.npy'.format(split, density_label)), 'wb'))
         #   wavenumber = result.poles.k
         #   multipoles = result.poles.power
            # pickles/save in format Auto_{Redshift or Real Space}_{Density Number}
    #        np.save(os.path.join(save_path, 
     #               'Auto_{0}_{1}_Wavenumber.npy'.format(split, density_label)),
     #               wavenumber)
    #        np.save(os.path.join(save_path,
     #               'Auto_{0}_{1}_Multipoles.npy'.format(split, density_label)),
     #               multipoles)
            
        # then does the cross correlation power spectrum of each quintile with
        # full field
        for i in range(len(density_position_list)):
            density_positions = density_position_list[i]
            density_label = density_labels[i]
          #  setup_logging()
            result = CatalogFFTPower(
                data_positions1=density_positions,
                data_positions2=all_positions,
                edges=np.linspace(0, 1, 151),
                boxsize=2000, cellsize=5, los='z', position_type='pos')
            pickle.dump(result, open(os.path.join(save_path, 'Cross_{0}_{1}.npy'.format(split, density_label)), 'wb'))
       #     wavenumber = result.poles.k
       #     multipoles = result.poles.power
            # save in format Cross_{Redshift or Real Space}_{Density Number}
       ##     np.save(os.path.join(save_path,
        #            'Cross_{0}_{1}_Wavenumber.npy'.format(split, density_label)),
       #             wavenumber)
        #    np.save(os.path.join(save_path,
        #            'Cross_{0}_{1}_Multipoles.npy'.format(split, density_label)),
        #            multipoles)