""" 
This script reads in the generated power spectra data for real and redshift
space positions for the AbacusSummit data, and creates plots.
"""


import pickle
import os
#import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    result_path = '/home/jgmorawe/results/power_spectra'
    
    # overlays real and redshift space power spectra for autocorrelations
    fig1, ax1 = plt.subplots(3,1,dpi=300)
    fig1.suptitle('Auto Correlation Power Spectra')
    density_labels = ['1', '2', '3', '4', '5']
    colors = ['red', 'darkorange', 'green', 'blue', 'indigo']
    for i in range(len(density_labels)):
        for split in ['z', 'r']:
            density_label = density_labels[i]
            color = colors[i]
            data = pickle.load(open(os.path.join(result_path, 'Auto_{0}_{1}.npy'.format(split, density_label)), 'rb'))
            data = data.poles
           # data = np.load(os.path.join(result_path, 'Auto_{0}_{1}.npy'.format(split, density_label)))
            wavenumber1, poles1 = data(ell=0, return_k=True, complex=False)
            wavenumber2, poles2 = data(ell=2, return_k=True, complex=False)
            wavenumber3, poles3 = data(ell=4, return_k=True, complex=False)
           # wavenumber = np.load(os.path.join(result_path, 'Auto_{0}_{1}.npy'.format(split, density_label)))
          #  poles = np.load(os.path.join(result_path, 'Auto_{0}_{1}_Multipoles.npy'.format(split, density_label)))
            if split == 'z':
                ax1[0].plot(wavenumber1, wavenumber1**2*poles1, '-', color=color, label='DS{}, Redshift Monopole'.format(int(density_label)))
                ax1[1].plot(wavenumber2, wavenumber2**2*poles2, '-', color=color, label='DS{}, Redshift Quadropole'.format(int(density_label)))
                ax1[2].plot(wavenumber3, wavenumber3**2*poles3, '-', color=color, label='DS{}, Redshift Hexapole'.format(int(density_label)))
            else:
                ax1[0].plot(wavenumber1, wavenumber1**2*poles1, '--', color=color, label='DS{}, Real Monopole'.format(int(density_label)))
                ax1[1].plot(wavenumber2, wavenumber2**2*poles2, '--', color=color, label='DS{}, Real Quadropole'.format(int(density_label)))
                ax1[2].plot(wavenumber3, wavenumber3**2*poles3, '--', color=color, label='DS{}, Real Hexapole'.format(int(density_label)))
    ax1[0].legend()
    fig1.savefig(os.path.join(result_path, 'Auto_Comparison_1.png'))
    
    
    # overlays real and redshift space power spectra for cross correlations (starts with just monopole)
    fig2, ax2 = plt.subplots(3,1,dpi=300)
    fig2.suptitle('Cross Correlation Power Spectra')
    density_labels = ['1', '2', '3', '4', '5']
    colors = ['red', 'darkorange', 'green', 'blue', 'indigo']
    for i in range(len(density_labels)):
        for split in ['z', 'r']:
            density_label = density_labels[i]
            color = colors[i]
            data = pickle.load(open(os.path.join(result_path, 'Cross_{0}_{1}.npy'.format(split, density_label)), 'rb'))
            data = data.poles
            wavenumber1, poles1 = data(ell=0, return_k=True, complex=False)
            wavenumber2, poles2 = data(ell=2, return_k=True, complex=False)
            wavenumber3, poles3 = data(ell=4, return_k=True, complex=False)
          #  data = np.load(os.path.join(result_path, 'Cross_{0}_{1}.npy'.format(split, density_label)))
          #  wavenumber = data.poles.k
          #  poles = data.poles.power
           # wavenumber = np.load(os.path.join(result_path, 'Auto_{0}_{1}.npy'.format(split, density_label)))
          #  poles = np.load(os.path.join(result_path, 'Auto_{0}_{1}_Multipoles.npy'.format(split, density_label)))
            if split == 'z':
                ax2[0].plot(wavenumber1, wavenumber1**2*poles1, '-', color=color, label='DS{}, Redshift Monopole'.format(int(density_label)))
                ax2[1].plot(wavenumber2, wavenumber2**2*poles2, '-', color=color, label='DS{}, Redshift Quadropole'.format(int(density_label)))
                ax2[2].plot(wavenumber3, wavenumber3**2*poles3, '-', color=color, label='DS{}, Redshift Hexapole'.format(int(density_label)))
            else:
                ax2[0].plot(wavenumber1, wavenumber1**2*poles1, '--', color=color, label='DS{}, Real Monopole'.format(int(density_label)))
                ax2[1].plot(wavenumber2, wavenumber2**2*poles2, '--', color=color, label='DS{}, Real Quadropole'.format(int(density_label)))
                ax2[2].plot(wavenumber3, wavenumber3**2*poles3, '--', color=color, label='DS{}, Real Hexapole'.format(int(density_label)))
    ax2[0].legend()
    fig2.savefig(os.path.join(result_path, 'Cross_Comparison_1.png'))
           # wavenumber = np.load(os.path.join(result_path, 'Cross_{0}_{1}_Wavenumber.npy'.format(split, density_label)))
           # poles = np.load(os.path.join(result_path, 'Cross_{0}_{1}_Multipoles.npy'.format(split, density_label)))
          #  if split == 'z':
          #      ax2.plot(wavenumber, np.abs(poles[0]), '-', color=color, label='DS{}, Redshift'.format(int(density_label)+1))
          #  else:
      #          ax2.plot(wavenumber, np.abs(poles[0]), '--', color=color, label='DS{}, Real'.format(int(density_label)+1))
  #  ax2.legend()
   # fig2.savefig(os.path.join(result_path, 'Cross_Comparison_1.png'))