"""
Generates autocorrelation functions for the halo field.
"""


import os
import time
import pickle
import numpy as np
from astropy.io import fits
from pycorr import TwoPointCorrelationFunction

import matplotlib.pyplot as plt



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
    
#    corr_folder_path = '/home/jgmorawe/results/correlation/monopole/real/halos'
    
 #   fig, ax = plt.subplots(dpi=500)
 #   
  #  corr_sum = np.zeros(28)
  #  n_sum = 0
    
  #  for sim_num in range(3000, 5000):
        
  #      path = os.path.join(corr_folder_path,
  #                          'sim{}_auto_halo.pkl'.format(sim_num))
  #      if os.path.exists(path):
  #          result = pickle.load(open(path, 'rb'))
  #          sep_vals = result.sep
  #          corr_vals = result.corr
  #          corr_sum += corr_vals
  #          n_sum += 1
  #          ax.plot(sep_vals, corr_vals, 'go', markersize=0.25)
    
#    corr_sum /= n_sum
 #   ax.plot(sep_vals, corr_sum, 'r-', linewidth=1)
  #  fig.savefig('/home/jgmorawe/results/plot_data/halo_correlations.png')
    
    t0 = time.time()
    
    # sets the relevant parameters
    boxsize = 500
    smooth_ds = 20
    los = 'z'
    split = 'r'
    
    sim_folder_path = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small'
    corr_folder_path = '/home/jgmorawe/results/correlation/monopole/real/halos'
    
    for sim_num in range(3000, 5000):
        
        sim_path = os.path.join(
            sim_folder_path,
            'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
            'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
        
        if os.path.exists(sim_path):
            
            halo_positions = get_data_positions(
                data_fn=sim_path, split=split, los=los)
            edges = np.linspace(10, 150, 29)
            auto = TwoPointCorrelationFunction(
                mode='s', edges=edges, data_positions1=halo_positions,
                position_type='pos', los=los, boxsize=boxsize)
            pickle.dump(
                auto, open(os.path.join(corr_folder_path, 
                'sim{}_auto_halo.pkl'.format(sim_num)), 'wb'))
        print('Job {} done.'.format(sim_num))
    
    print('Script executed in {} seconds.'.format(time.time()-t0))
        
        
        
        
        
        
        
    
