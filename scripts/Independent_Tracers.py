# Plots the probability distribution of bias ratios, depending on whether
# the measurements are made on independent tracers or not

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

#power_path = '/home/jgmorawe/results/power_spectra/real'
correlation_path = '/home/jgmorawe/results/correlation'


fig, ax = plt.subplots(dpi=300)

b1b5_dependent = []
b1b5_independent = []

# runs dependent tracers first
for sim_num in np.arange(3000, 4999):
    try:
        s = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).sep
        corr1 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).corr
        corr5 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross5.pkl'.format(sim_num)), 'rb')).corr    
        b1b5 = np.mean(corr1/corr5)
        b1b5_dependent.append(b1b5)
    except:
        continue

# runs independent tracers next
for sim_num in np.arange(3000, 4999):
    try:
        s = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).sep
        corr1 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).corr
        corr5 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross5.pkl'.format(sim_num+1)), 'rb')).corr    
        b1b5 = np.mean(corr1/corr5)
        b1b5_independent.append(b1b5)
    except:
        continue
        
fig, ax = plt.subplots(2,1, sharex=True, dpi=300)
ax[0].hist(b1b5_dependent, color='red')
ax[1].hist(b1b5_independent, color='green')
fig.savefig('/home/jgmorawe/results/plot_data/independent_bias_ratios.png')

#for sim_num in np.arange(3000, 5000):#[3788, 3172, 4159, 4191, 4435]:
  #  try:
   #     s = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).sep
    #    corr1 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).corr
   #     corr5 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross5.pkl'.format(sim_num)), 'rb')).corr    
   #     b1b5 = np.mean(corr1/corr5)
        #cross1 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).corr
        #cross5 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross5.pkl'.format(sim_num)), 'rb')).corr
        ##s, corr1 = cross1(ells=(0), return_sep=True)
        #s, corr5 = cross5(ells=(0), return_sep=True)
       # cross1 = pickle.load(open(os.path.join(power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num)), 'rb')).poles
       # cross5 = pickle.load(open(os.path.join(power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num)), 'rb')).poles
       # s, #k = cross1(ell=(0), return_k=True, complex=False)[0]
        #pow1 = cross1(ell=(0), return_k=False, complex=False)
        #pow5 = cross5(ell=(0), return_k=False, complex=False)
        #b1b5 = pow1/pow5
    #    ax.plot(s, s**2*corr1, 'o', markersize=1, color='red')
    #    ax.plot(s, s**2*corr5, 'o', markersize=1, color='blue')
   # except:
   #     continue

#fig.savefig('/home/jgmorawe/results/plot_data/independent_ratios.png')    
