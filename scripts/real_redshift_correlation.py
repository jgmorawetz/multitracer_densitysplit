"""
This script is used to generate auto and cross correlation for the real and
redshift space positions for the AbacusSummit data.
"""


import os
import numpy as np
from pycorr import TwoPointCorrelationFunction


if __name__ == '__main__':
    # repeats procedure for both 'z' (redshift space) and 'r' (real space)
    for split in ['z', 'r']:
        # reads in quintile data positions
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
        # combines quintiles to also give a catalog of all positions combined
        all_positions = np.vstack((density_1_positions, density_2_positions,
                                   density_3_positions, density_4_positions,
                                   density_5_positions))
        density_position_list = [density_1_positions, density_2_positions,
                                 density_3_positions, density_4_positions,
                                 density_5_positions]
        del(density_1_positions, density_2_positions, density_3_positions,
            density_4_positions, density_5_positions)
        density_labels = ['1', '2', '3', '4', '5']
        save_path = '/home/jgmorawe/results/correlation'
        # first computes the auto correlation for each quintile's positions
        for i in range(len(density_position_list)):
            density_positions = density_position_list[i]
            density_label = density_labels[i]
            result = TwoPointCorrelationFunction(
                        mode='smu', 
                        edges=(np.linspace(0, 150, 151), np.linspace(-1, 1, 241)),
                        data_positions1=density_positions,
                        position_type='pos', los='z', boxsize=2000)
            separation = result.sep
            correlation = result.corr
            # save in format Auto_{Redshift or Real Space}_{Density Number}
            np.save(os.path.join(save_path,
                    'Auto_{0}_{1}_Separation.npy'.format(split, density_label)),
                    separation)
            np.save(os.path.join(save_path,
                    'Auto_{0}_{1}_Correlation.npy'.format(split, density_label)),
                    correlation)
        
        # then does cross correlation for each quintile with the full field
        for i in range(len(density_position_list)):
            density_positions = density_position_list[i]
            density_label = density_labels[i]
            result = TwoPointCorrelationFunction(
                        mode='smu',
                        edges=(np.linspace(0, 150, 151), np.linspace(-1, 1, 241)),
                        data_positions1 = density_positions,
                        data_positions2 = all_positions,
                        position_type='pos', los='z', boxsize=2000)
            separation=result.sep
            correlation=result.corr
            # save in format Cross_{Redshift or Real Space}_{Density Number}
            np.save(os.path.join(save_path,
                    'Cross_{0}_{1}_Separation.npy'.format(split, density_label)),
                    separation)
            np.save(os.path.join(save_path,
                    'Cross_{0}_{1}_Correlation.npy'.format(split, density_label)),
                    correlation)