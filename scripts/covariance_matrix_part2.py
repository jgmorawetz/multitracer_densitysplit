"""
This script reads in the previously created power spectra results from the 
different AbacusSummit simulation realizations, and generates the resulting
covariance matrix.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

if __name__ == '__main__':

    power_path = '/home/jgmorawe/results/power_spectra'
    matrix_path = '/home/jgmorawe/results/plot_data'
    
    # for real and redshift space separately, reads in the generated 
    # power spectrum values and generates a data vector (each data vector
    # has all the power spectrum values, across all 5 quintiles; more 
    # specifically there are 50 bins and 5 quintiles so 250 total elements)
    # performs for cross correlation first (might do autocorrelation later)
    
    # computes the number of data vectors to initiate array to store
    # data vectors 
    n_data = len(os.listdir(power_path))
    # computes dimensions of array
    n_row = 31 * 2#5
    n_col = n_data
    data_vectors = np.zeros((n_row, n_col))
    
    sim_nums = list(map(lambda x: int(x.split('.')[0].split('sim')[-1]), os.listdir(power_path)))
    col_ind = 0
    for sim_num in sim_nums:
        result1 = pickle.load(open(os.path.join(power_path, 'Cross_r_1_sim{}.pkl'.format(sim_num)), 'rb')).poles
        result2 = pickle.load(open(os.path.join(power_path, 'Cross_r_2_sim{}.pkl'.format(sim_num)), 'rb')).poles
        result3 = pickle.load(open(os.path.join(power_path, 'Cross_r_3_sim{}.pkl'.format(sim_num)), 'rb')).poles
        result4 = pickle.load(open(os.path.join(power_path, 'Cross_r_4_sim{}.pkl'.format(sim_num)), 'rb')).poles
        result5 = pickle.load(open(os.path.join(power_path, 'Cross_r_5_sim{}.pkl'.format(sim_num)), 'rb')).poles
        k = result1(ell=[0], return_k=True, complex=False)[0][:31]
        pow1 = result1(ell=[0], return_k=False, complex=False)[0][:31]*k**2 # since some are unexpectedly nan
        pow2 = result2(ell=[0], return_k=False, complex=False)[0][:31]*k**2
        pow3 = result3(ell=[0], return_k=False, complex=False)[0][:31]*k**2
        pow4 = result4(ell=[0], return_k=False, complex=False)[0][:31]*k**2
        pow5 = result5(ell=[0], return_k=False, complex=False)[0][:31]*k**2
        full_pow = np.concatenate((pow1, pow5))#np.concatenate((pow1, pow2, pow3, pow4, pow5))
        data_vectors[:, col_ind] = full_pow
        col_ind += 1
        
    covariance_matrix = np.cov(data_vectors)
    np.save(os.path.join(matrix_path, 'Cross_Real_Monopole_Covariance.npy'), covariance_matrix)
    
    fig, ax = plt.subplots(dpi=500)
    #pcm = ax.pcolormesh(np.arange(0, np.shape(covariance_matrix)[0]), np.arange(0, np.shape(covariance_matrix)[0]), covariance_matrix,
    #                   norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
    #                                          vmin=-1.0, vmax=1.0, base=10),
    #                   cmap='RdBu_r', shading='auto')
   # fig.colorbar(pcm, ax=ax[0], extend='both')
    ax.imshow(covariance_matrix)
    fig.savefig(os.path.join(matrix_path,'Covariance_Matrix_Plot.png'))
    
    
    
    