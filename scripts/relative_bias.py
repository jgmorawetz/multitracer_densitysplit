"""
Reads in the created cross power spectra, and finds the relative bias
distribution for the same and independent realizations separately.
"""



import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



power_path = '/home/jgmorawe/results/power_spectra/real'
plot_path = '/home/jgmorawe/results/plot_data/bias_ratios'



################## DEPENDENT TRACERS (FROM SAME REALIZATION) ##################

# counts the number of simulations first (cannot have any nans)
valid_sims = []
for sim_num in np.arange(3000, 5000):
    sim_path1 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num))
    sim_path5 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num))
    if os.path.exists(sim_path1) and os.path.exists(sim_path5):
        pow1 = pickle.load(open(sim_path1, 'rb')).poles.power[0].real
        pow5 = pickle.load(open(sim_path5, 'rb')).poles.power[0].real
        ratio = np.array(pow1/pow5)
        flag = np.sum(ratio)
        if np.isnan(flag):
            continue
        valid_sims.append(sim_num)
n_sims = len(valid_sims); print('Valid simulations = {}'.format(n_sims))
        
# computes for covariance matrix for P1(k)/P5(k) across all simulations
observed_ratios = []
for sim_num in valid_sims:
    sim_path1 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num))
    sim_path5 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num))
    pow1 = pickle.load(open(sim_path1, 'rb')).poles.power[0].real
    pow5 = pickle.load(open(sim_path5, 'rb')).poles.power[0].real
    ratio = np.array(pow1/pow5)
    observed_ratios.append(ratio)
observed_ratios = np.array(observed_ratios).T
cov_mat = np.cov(observed_ratios)
inv_cov_mat = np.linalg.inv(cov_mat)
print(inv_cov_mat)##############
dim = np.shape(inv_cov_mat)[0]; print(dim)
print(dim)##################

# computes the best fit value of relative bias b15
dependent_biases = []
one_vec_trans = np.array([np.ones(dim)])
one_vec = one_vec_trans.T
for sim_num in valid_sims:
    sim_path1 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num))
    sim_path5 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num))
    pow1 = pickle.load(open(sim_path1, 'rb')).poles.power[0].real
    pow5 = pickle.load(open(sim_path5, 'rb')).poles.power[0].real
    ratio = np.array(pow1/pow5)
    ratio_vec_trans = np.array([ratio])
    ratio_vec = ratio_vec_trans.T
    bias = ((np.matmul(np.matmul(ratio_vec_trans, inv_cov_mat), one_vec) + 
             np.matmul(np.matmul(one_vec_trans, inv_cov_mat), ratio_vec))/
            (2*np.matmul(np.matmul(one_vec_trans, inv_cov_mat), one_vec)))
    dependent_biases.append(bias[0][0])
#print(dependent_biases)
print('Dependent Bias done.')



############## INDEPENDENT TRACERS (FROM SEPARATE REALIZATIONS) ###############

# counts number of simulation pairs (since using different realizations now)
valid_sim_pairs = []
for sim_num in np.arange(3000, 4999):
    sim_path1 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num))
    sim_path5 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num+1))
    if os.path.exists(sim_path1) and os.path.exists(sim_path5):
        pow1 = pickle.load(open(sim_path1, 'rb')).poles.power[0].real
        pow5 = pickle.load(open(sim_path5, 'rb')).poles.power[0].real
        ratio = np.array(pow1/pow5)
        flag = np.sum(ratio)
        if np.isnan(flag):
            continue
        valid_sim_pairs.append((sim_num, sim_num+1))
n_sim_pairs = len(valid_sim_pairs); print('Valid sim pairs = {}'.format(n_sim_pairs))

# computes the covariance matrix for P1i(k)/P5j(k) across each simulation pair
observed_ratios = []
for sim_pair in valid_sim_pairs:
    sim_num1 = sim_pair[0]
    sim_num5 = sim_pair[-1]
    sim_path1 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num1))
    sim_path5 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num5))
    pow1 = pickle.load(open(sim_path1, 'rb')).poles.power[0].real
    pow5 = pickle.load(open(sim_path5, 'rb')).poles.power[0].real
    ratio = np.array(pow1/pow5)
    observed_ratios.append(ratio)
observed_ratios = np.array(observed_ratios).T
cov_mat = np.cov(observed_ratios)
inv_cov_mat = np.linalg.inv(cov_mat)
print(inv_cov_mat)###############
dim = np.shape(inv_cov_mat)[0]
print(dim)#####################
print(dim)######################

# computes best fit value of relative bias b15 (for separate realizations)
independent_biases = []
one_vec_trans = np.array([np.ones(dim)])
one_vec = one_vec_trans.T
for sim_pair in valid_sim_pairs:
    sim_num1 = sim_pair[0]
    sim_num5 = sim_pair[-1]
    sim_path1 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds1.pkl'.format(sim_num1))
    sim_path5 = os.path.join(
        power_path, 'sim{}_monopole_cross_ds5.pkl'.format(sim_num5))
    pow1 = pickle.load(open(sim_path1, 'rb')).poles.power[0].real
    pow5 = pickle.load(open(sim_path5, 'rb')).poles.power[0].real
    ratio = np.array(pow1/pow5)
    ratio_vec_trans = np.array([ratio])
    ratio_vec = ratio_vec_trans.T
    bias = ((np.matmul(np.matmul(ratio_vec_trans, inv_cov_mat), one_vec) + 
             np.matmul(np.matmul(one_vec_trans, inv_cov_mat), ratio_vec))/
            (2*np.matmul(np.matmul(one_vec_trans, inv_cov_mat), one_vec)))
    independent_biases.append(bias[0][0])
print('Independent Bias done.')


# PLOTS BOTH RESULTS
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300, figsize=(5,8))
fig.suptitle('Bias Ratio b1/b5 Distribution', weight='bold')
ax[0].set_title('Same Realization')
ax[1].set_title('Independent Realization')
ax[0].hist(dependent_biases, bins=np.arange(-0.7, -0.5, 0.001))#, bins=50)
ax[1].hist(independent_biases, bins=np.arange(-0.7, -0.5, 0.001))
fig.savefig(os.path.join(plot_path, 'bias_ratio_histogram.png'))