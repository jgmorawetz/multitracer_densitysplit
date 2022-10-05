"""
This script is used to generate auto and cross correlation for the real and
redshift space positions for the AbacusSummit data.
"""


import os
from astropy.io import fits
import pickle
import numpy as np
from pycorr import TwoPointCorrelationFunction

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
        # reads in quintile data positions
        quintile_path = os.path.join(
            '/home/jgmorawe/results/quintiles/',
            'quantiles_{}split_base_c000_ph000_z0.575_nden3.2e-04.npy'.format(
              split))
        quintile_data = np.load(quintile_path)
        ds1_positions = quintile_data[0]
        ds2_positions = quintile_data[1]
        ds3_positions = quintile_data[2]
        ds4_positions = quintile_data[3]
        ds5_positions = quintile_data[4]
        del(quintile_data)
        # reads in the 'actual' galaxy positions to cross correlate with
        # random quintiles
        data_dir = os.path.join('/home/jgmorawe/projects/rrg-wperciva/',
                                'AbacusSummit/AbacusSummit_base_c000_ph000/',
                                'halos/z0.575')
        # uses cosmo = 0, phase = 0, redshift = 0.575 (may need to adjust)
        data_fn = os.path.join(data_dir,
                               'halos_base_c000_ph000_z0.575_nden3.2e-04.fits')
        all_positions = get_data_positions(data_fn=data_fn, split=split, los='z')
        density_position_list = [ds1_positions, ds2_positions,
                                 ds3_positions, ds4_positions,
                                 ds5_positions]
        del(ds1_positions, ds2_positions, ds3_positions,
            ds4_positions, ds5_positions)
        density_labels = ['1', '2', '3', '4', '5']
        save_path = '/home/jgmorawe/results/correlation'
        # first computes the auto correlation for each quintile's positions
        for i in range(len(density_position_list)):
            ds_positions = density_position_list[i]
            density_label = density_labels[i]
            result = TwoPointCorrelationFunction(
                        mode='smu', 
                        edges=(np.linspace(0, 150, 151), np.linspace(-1, 1, 241)),
                        data_positions1=ds_positions,
                        position_type='pos', los='z', boxsize=2000)
            # saves the 'result' as a pickle file to be able to read in attributes
            # later without having to specify them now
            pickle.dump(result, open(os.path.join(save_path, 
                        'Auto_{0}_{1}.pkl'.format(split, density_label)), 'wb'))

        # then does cross correlation for each quintile with the full field
        for i in range(len(density_position_list)):
            ds_positions = density_position_list[i]
            density_label = density_labels[i]
            result = TwoPointCorrelationFunction(
                        mode='smu',
                        edges=(np.linspace(0, 150, 151), np.linspace(-1, 1, 241)),
                        data_positions1 = ds_positions,
                        data_positions2 = all_positions,
                        position_type='pos', los='z', boxsize=2000)
            # saves the 'result' as a pickle file to be able to read in attributes
            # later without having to specify them now
            pickle.dump(result, open(os.path.join(save_path,
                        'Cross_{0}_{1}.pkl'.format(split, density_label)), 'wb'))