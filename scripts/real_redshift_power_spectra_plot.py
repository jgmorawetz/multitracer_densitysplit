""" 
This script reads in the generated power spectra data for real and redshift
space positions for the AbacusSummit data, and creates plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    result_path = '/home/jgmorawe/results/power_spectra'
    
    # overlays real and redshift space power spectra for autocorrelations (starts with just monopole)
    fig1, ax1 = plt.subplots(dpi=300)
    fig1.suptitle('Auto Correlation Power Spectra (Monopoles)')
    density_labels = ['1', '2', '3', '4', '5']
    colors = ['red', 'darkorange', 'green', 'blue', 'indigo']
    for i in range(len(density_labels)):
        for split in ['z', 'r']:
            density_label = density_labels[i]
            color = colors[i]
            wavenumber = np.load(os.path.join(result_path, 'Auto_{0}_{1}_Wavenumber.npy'.format(split, density_label)))
            poles = np.load(os.path.join(result_path, 'Auto_{0}_{1}_Multipoles.npy'.format(split, density_label)))
            if split == 'z':
                ax1.plot(wavenumber, np.abs(poles[0]), '-', color=color, label='DS{}, Redshift'.format(int(density_label)+1))
            else:
                ax1.plot(wavenumber, np.abs(poles[0]), '--', color=color, label='DS{}, Real'.format(int(density_label)+1))
    ax1.legend()
    fig1.savefig(os.path.join(result_path, 'Auto_Comparison.png'))
    
    
    # overlays real and redshift space power spectra for cross correlations (starts with just monopole)
    fig2, ax2 = plt.subplots(dpi=300)
    fig2.suptitle('Cross Correlation Power Spectra (Monopoles)')
    density_labels = ['1', '2', '3', '4', '5']
    colors = ['red', 'darkorange', 'green', 'blue', 'indigo']
    for i in range(len(density_labels)):
        for split in ['z', 'r']:
            density_label = density_labels[i]
            color = colors[i]
            wavenumber = np.load(os.path.join(result_path, 'Cross_{0}_{1}_Wavenumber.npy'.format(split, density_label)))
            poles = np.load(os.path.join(result_path, 'Cross_{0}_{1}_Multipoles.npy'.format(split, density_label)))
            if split == 'z':
                ax2.plot(wavenumber, np.abs(poles[0]), '-', color=color, label='DS{}, Redshift'.format(int(density_label)+1))
            else:
                ax2.plot(wavenumber, np.abs(poles[0]), '--', color=color, label='DS{}, Real'.format(int(density_label)+1))
    ax2.legend()
    fig2.savefig(os.path.join(result_path, 'Cross_Comparison.png'))