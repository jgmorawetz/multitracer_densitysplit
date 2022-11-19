""" 
Determines the relative bias for dependent and independent realization.
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    
    
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, dpi=300, figsize=(5,8))
    
    up_ind = 6
    
    # first determines the valid dependent simulation pairs
    dep_pairs = []
    for sim_num in range(3000, 5000):
        path1 = os.path.join('/home/jgmorawe/results/power_spectra/monopole/real',
                                       'sim{}_cross_ds1halo.pkl'.format(sim_num))
        path5 = os.path.join('/home/jgmorawe/results/power_spectra/monopole/real',
                                       'sim{}_cross_ds5halo.pkl'.format(sim_num))
        if os.path.exists(path1) and os.path.exists(path5):
            DS1 = pickle.load(open(path1, 'rb'))
            DS5 = pickle.load(open(path5, 'rb'))
            d1 = np.array(DS1.poles.power[0].real[1:up_ind])
            d5 = np.array(DS5.poles.power[0].real[1:up_ind])
            ratio = np.array(d1/d5)
            flag = np.sum(ratio)
            if np.isnan(flag):
                continue
            dep_pairs.append((sim_num, sim_num))
    
    # next determines the valid independent simulation pairs
    indep_pairs = []
    for sim_num in range(3000, 4999):
        path1 = os.path.join('/home/jgmorawe/results/power_spectra/monopole/real',
                                       'sim{}_cross_ds1halo.pkl'.format(sim_num))
        path5 = os.path.join('/home/jgmorawe/results/power_spectra/monopole/real',
                                       'sim{}_cross_ds5halo.pkl'.format(sim_num+1))
        if os.path.exists(path1) and os.path.exists(path5):
            DS1 = pickle.load(open(path1, 'rb'))
            DS5 = pickle.load(open(path5, 'rb'))
            d1 = np.array(DS1.poles.power[0].real[1:up_ind])
            d5 = np.array(DS5.poles.power[0].real[1:up_ind])
            ratio = np.array(d1/d5)
            flag = np.sum(ratio)
            if np.isnan(flag):
                continue
            indep_pairs.append((sim_num, sim_num+1))
    
    # first estimates the approximate relative bias for purpose of calculating the covariance matrices
    b15 = -0.8 # vary this as needed
    
    # determines the dependent covariance matrix (using a pre-estimated relative bias value)
    dep_errs = []
    for sim_pair in dep_pairs:
        path1 = os.path.join(
            '/home/jgmorawe/results/power_spectra/monopole/real',
            'sim{}_cross_ds1halo.pkl'.format(sim_pair[0]))
        path5 = os.path.join(
            '/home/jgmorawe/results/power_spectra/monopole/real',
            'sim{}_cross_ds5halo.pkl'.format(sim_pair[1]))
        if os.path.exists(path1) and os.path.exists(path5):
            DS1 = pickle.load(open(path1, 'rb'))
            DS5 = pickle.load(open(path5, 'rb'))
            k = DS1.poles.k[1:]
            d1 = np.array(DS1.poles.power[0].real[1:up_ind])
            d5 = np.array(DS5.poles.power[0].real[1:up_ind])
            dep_errs.append(d1-b15*d5)
    C_dep = np.cov(np.array(dep_errs).T)
    i_C_dep = np.linalg.inv(C_dep)
    
    
    # determines the independent covariance matrix (using a pre-estimated relative bias value)
    indep_errs = []
    for sim_pair in indep_pairs:
        path1 = os.path.join(
            '/home/jgmorawe/results/power_spectra/monopole/real',
            'sim{}_cross_ds1halo.pkl'.format(sim_pair[0]))
        path5 = os.path.join(
            '/home/jgmorawe/results/power_spectra/monopole/real',
            'sim{}_cross_ds5halo.pkl'.format(sim_pair[1]))
        if os.path.exists(path1) and os.path.exists(path5):
            DS1 = pickle.load(open(path1, 'rb'))
            DS5 = pickle.load(open(path5, 'rb'))
            k = DS1.poles.k[1:]
            d1 = np.array(DS1.poles.power[0].real[1:up_ind])
            d5 = np.array(DS5.poles.power[0].real[1:up_ind])
            indep_errs.append(d1-b15*d5)
    C_indep = np.cov(np.array(indep_errs).T)
    i_C_indep = np.linalg.inv(C_indep)
    
    
    # now computes the b15 values for the dependent realizations
    b15_dep = []
    for sim_pair in dep_pairs:
        path1 = os.path.join('/home/jgmorawe/results/power_spectra/monopole/real',
                                       'sim{}_cross_ds1halo.pkl'.format(sim_pair[0]))
        path5 = os.path.join('/home/jgmorawe/results/power_spectra/monopole/real',
                                       'sim{}_cross_ds5halo.pkl'.format(sim_pair[1]))
        if os.path.exists(path1) and os.path.exists(path5):
            DS1 = pickle.load(open(path1, 'rb'))
            DS5 = pickle.load(open(path5, 'rb'))
            k = DS1.poles.k[1:]
            d1 = np.array([DS1.poles.power[0].real[1:up_ind]]).T
            d5 = np.array([DS5.poles.power[0].real[1:up_ind]]).T
            d1t = np.transpose(d1)
            d5t = np.transpose(d5)
         #   print(d1t)
          #  print(np.shape(d1t), np.shape(i_C_dep), np.shape(d5))
            b15_dep_now = ((np.matmul(np.matmul(d1t, i_C_dep), d5) + 
                           np.matmul(np.matmul(d5t, i_C_dep), d1))/
                          (2*np.matmul(np.matmul(d5t, i_C_dep), d5)))
            b15_dep.append(b15_dep_now[0][0])
    
    ax[0].hist(b15_dep, bins=30)
            
    # now computes the b15 values for the independent realizations
    b15_indep = []        
    for sim_pair in indep_pairs:
        path1 = os.path.join(
            '/home/jgmorawe/results/power_spectra/monopole/real',
            'sim{}_cross_ds1halo.pkl'.format(sim_pair[0]))
        path5 = os.path.join(
            '/home/jgmorawe/results/power_spectra/monopole/real',
            'sim{}_cross_ds5halo.pkl'.format(sim_pair[1]))
        if os.path.exists(path1) and os.path.exists(path5):
            DS1 = pickle.load(open(path1, 'rb'))
            DS5 = pickle.load(open(path5, 'rb'))
            k = DS1.poles.k[1:]
            d1 = np.array([DS1.poles.power[0].real[1:up_ind]]).T
            d5 = np.array([DS5.poles.power[0].real[1:up_ind]]).T
            d1t = np.transpose(d1)
            d5t = np.transpose(d5)
            b15_indep_now = ((np.matmul(np.matmul(d1t, i_C_indep), d5) + 
                             np.matmul(np.matmul(d5t, i_C_indep), d1))/
                            (2*np.matmul(np.matmul(d5t, i_C_indep), d5)))
            b15_indep.append(b15_indep_now[0][0])
        
        
    ax[1].hist(b15_indep, bins=30)
    fig.savefig('/home/jgmorawe/results/plot_data/bias_histogram_crosspower.png')
    print(np.std(b15_dep)); print(np.mean(b15_dep))
    print(np.std(b15_indep)); print(np.mean(b15_indep))