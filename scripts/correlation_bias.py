import os
import numpy as np
import matplotlib.pyplot as plt
data_path = '/Users/jamesmorawetz/Documents/correlation/data/ds'



dim_corr = 28
sep_edges = np.linspace(10, 150, dim_corr+1)
sep = np.array([np.mean(sep_edges[i:i+2]) for i in range(len(sep_edges)-1)])



# generates arrays which contain all the realizations of the different correlation functions
# (the rows are separate realizations, while columns are separate variables)
auto_ds1 = []
auto_ds5 = []
auto_halo = []
cross_ds1_halo = []
cross_ds5_halo = []
cross_ds1_ds5 = []
n_realizations = 0
for sim_num in range(3000, 5000):
    test_path = os.path.join(data_path, 'sim{}_auto_ds1_ds1_corr.txt'.format(sim_num))
    if os.path.exists(test_path):
        auto_ds1_now = np.loadtxt(os.path.join(data_path, 'sim{}_auto_ds1_ds1_corr.txt'.format(sim_num)))
        auto_ds5_now = np.loadtxt(os.path.join(data_path, 'sim{}_auto_ds5_ds5_corr.txt'.format(sim_num)))
        auto_halo_now = np.loadtxt(os.path.join(data_path, 'sim{}_auto_halo_corr.txt'.format(sim_num)))
        cross_ds1_halo_now = np.loadtxt(os.path.join(data_path, 'sim{}_cross_ds1_halo_corr.txt'.format(sim_num)))
        cross_ds5_halo_now = np.loadtxt(os.path.join(data_path, 'sim{}_cross_ds5_halo_corr.txt'.format(sim_num)))
        cross_ds1_ds5_now = np.loadtxt(os.path.join(data_path, 'sim{}_cross_ds1_ds5_corr.txt'.format(sim_num)))
        auto_ds1.append(auto_ds1_now)
        auto_ds5.append(auto_ds5_now)
        auto_halo.append(auto_halo_now)
        cross_ds1_halo.append(cross_ds1_halo_now)
        cross_ds5_halo.append(cross_ds5_halo_now)
        cross_ds1_ds5.append(cross_ds1_ds5_now)
        n_realizations += 1
auto_ds1 = np.array(auto_ds1)
auto_ds5 = np.array(auto_ds5)
auto_halo = np.array(auto_halo)
cross_ds1_halo = np.array(cross_ds1_halo)
cross_ds5_halo = np.array(cross_ds5_halo)
cross_ds1_ds5 = np.array(cross_ds1_ds5)
# generates the averages of each correlation function
auto_ds1_avg = np.zeros(dim_corr)
auto_ds5_avg = np.zeros(dim_corr)
auto_halo_avg = np.zeros(dim_corr)
cross_ds1_halo_avg = np.zeros(dim_corr)
cross_ds5_halo_avg = np.zeros(dim_corr)
cross_ds1_ds5_avg = np.zeros(dim_corr)
for i in range(n_realizations):
    auto_ds1_avg += auto_ds1[i]
    auto_ds5_avg += auto_ds5[i]
    auto_halo_avg += auto_halo[i]
    cross_ds1_halo_avg += cross_ds1_halo[i]
    cross_ds5_halo_avg += cross_ds5_halo[i]
    cross_ds1_ds5_avg += cross_ds1_ds5[i]
auto_ds1_avg /= n_realizations
auto_ds5_avg /= n_realizations
auto_halo_avg /= n_realizations
cross_ds1_halo_avg /= n_realizations
cross_ds5_halo_avg /= n_realizations
cross_ds1_ds5_avg /= n_realizations
# generates arrays which contain all the realizations of the different correlation function residuals
auto_ds1_res = auto_ds1.copy()
auto_ds5_res = auto_ds5.copy()
auto_halo_res = auto_halo.copy()
cross_ds1_halo_res = cross_ds1_halo.copy()
cross_ds5_halo_res = cross_ds5_halo.copy()
cross_ds1_ds5_res = cross_ds1_ds5.copy()
for i in range(len(auto_ds1_res)):
    auto_ds1_res[i] -= auto_ds1_avg
    auto_ds5_res[i] -= auto_ds5_avg
    auto_halo_res[i] -= auto_halo_avg
    cross_ds1_halo_res[i] -= cross_ds1_halo_avg
    cross_ds5_halo_res[i] -= cross_ds5_halo_avg
    cross_ds1_ds5_res[i] -= cross_ds1_ds5_avg
# generates the standard deviations of each correlation function
auto_ds1_err = np.zeros(dim_corr)
auto_ds5_err = np.zeros(dim_corr)
auto_halo_err = np.zeros(dim_corr)
cross_ds1_halo_err = np.zeros(dim_corr)
cross_ds5_halo_err = np.zeros(dim_corr)
cross_ds1_ds5_err = np.zeros(dim_corr)
for i in range(n_realizations):
    auto_ds1_err += (auto_ds1[i]-auto_ds1_avg)**2
    auto_ds5_err += (auto_ds5[i]-auto_ds5_avg)**2
    auto_halo_err += (auto_halo[i]-auto_halo_avg)**2
    cross_ds1_halo_err += (cross_ds1_halo[i]-cross_ds1_halo_avg)**2
    cross_ds5_halo_err += (cross_ds5_halo[i]-cross_ds5_halo_avg)**2
    cross_ds1_ds5_err += (cross_ds1_ds5[i]-cross_ds1_ds5_avg)**2
auto_ds1_err /= n_realizations
auto_ds5_err /= n_realizations
auto_halo_err /= n_realizations
cross_ds1_halo_err /= n_realizations
cross_ds5_halo_err /= n_realizations
cross_ds1_ds5_err /= n_realizations
auto_ds1_err = np.sqrt(auto_ds1_err)
auto_ds5_err = np.sqrt(auto_ds5_err)
auto_halo_err = np.sqrt(auto_halo_err)
cross_ds1_halo_err = np.sqrt(cross_ds1_halo_err)
cross_ds5_halo_err = np.sqrt(cross_ds5_halo_err)
cross_ds1_ds5_err = np.sqrt(cross_ds1_ds5_err)
    


# first makes a plot of the average correlation functions with their associated errors
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, 
                       figsize=(5,7), dpi=500)
fig.subplots_adjust(hspace=0.15)
ax[0].plot(sep, sep**2*auto_ds1_avg, '-', label='DS1-DS1', color='tab:red')
ax[0].plot(sep, sep**2*auto_ds5_avg, '-', label='DS5-DS5', color='tab:green')
ax[0].plot(sep, sep**2*auto_halo_avg, '-', label='Halo-Halo', color='tab:blue')
ax[0].fill_between(x=sep, y1=sep**2*(auto_ds1_avg-auto_ds1_err),
                   y2=sep**2*(auto_ds1_avg+auto_ds1_err), color='tab:red', alpha=0.1)
ax[0].fill_between(x=sep, y1=sep**2*(auto_ds5_avg-auto_ds5_err),
                   y2=sep**2*(auto_ds5_avg+auto_ds5_err), color='tab:green', alpha=0.1)
ax[0].fill_between(x=sep, y1=sep**2*(auto_halo_avg-auto_halo_err),
                   y2=sep**2*(auto_halo_avg+auto_halo_err), color='tab:blue', alpha=0.1)
ax[0].legend()
ax[0].set_xlim(0, 160)
ax[0].axhline(y=0, linewidth=0.5, linestyle='--', color='black')
ax[0].set_ylabel(r'$s^2 \epsilon(s) \ [h^{-2}Mpc^2]$')
ax[0].ticklabel_format(scilimits=(0,0), axis='y')
ax[0].set_title('Auto Correlations')
ax[1].plot(sep, sep**2*cross_ds1_halo_avg, '-', label='DS1-Halo', color='tab:red')
ax[1].plot(sep, sep**2*cross_ds5_halo_avg, '-', label='DS5-Halo', color='tab:green')
ax[1].plot(sep, sep**2*cross_ds1_ds5_avg, '-', label='DS1-DS5', color='tab:blue')
ax[1].fill_between(x=sep, y1=sep**2*(cross_ds1_halo_avg-cross_ds1_halo_err),
                   y2=sep**2*(cross_ds1_halo_avg+cross_ds1_halo_err), color='tab:red', alpha=0.1)
ax[1].fill_between(x=sep, y1=sep**2*(cross_ds5_halo_avg-cross_ds5_halo_err),
                   y2=sep**2*(cross_ds5_halo_avg+cross_ds5_halo_err), color='tab:green', alpha=0.1)
ax[1].fill_between(x=sep, y1=sep**2*(cross_ds1_ds5_avg-cross_ds1_ds5_err),
                   y2=sep**2*(cross_ds1_ds5_avg+cross_ds1_ds5_err), color='tab:blue', alpha=0.1)
ax[1].legend()
ax[1].axhline(y=0, linewidth=0.5, linestyle='--', color='black')
ax[1].set_xlabel(r'$s \ [h^{-1}Mpc]$')
ax[1].set_ylabel(r'$s^2 \epsilon(s) \ [h^{-2}Mpc^2]$')
ax[1].ticklabel_format(scilimits=(0,0), axis='y')
ax[1].set_title('Cross Correlations')



# computes the correlation coefficient between residuals (errors) between the different correlation function errors
corrcoef_auto15_dep = np.zeros(dim_corr)
corrcoef_auto15_indep = np.zeros(dim_corr)
corrcoef_crosshalo15_dep = np.zeros(dim_corr)
corrcoef_crosshalo15_indep = np.zeros(dim_corr)
corrcoef_cross15auto1_dep = np.zeros(dim_corr)
corrcoef_cross15auto1_indep = np.zeros(dim_corr)
corrcoef_cross15auto5_dep = np.zeros(dim_corr)
corrcoef_cross15auto5_indep = np.zeros(dim_corr)
for i in range(dim_corr):
    corrcoef_auto15_dep[i] = np.corrcoef(np.vstack((auto_ds1_res[:, i], auto_ds5_res[:, i])))[0][1]
    corrcoef_auto15_indep[i] = np.corrcoef(np.vstack((auto_ds1_res[:-1, i], auto_ds5_res[1:, i])))[0][1]
    corrcoef_crosshalo15_dep[i] = np.corrcoef(np.vstack((cross_ds1_halo_res[:, i], cross_ds5_halo_res[:, i])))[0][1]
    corrcoef_crosshalo15_indep[i] = np.corrcoef(np.vstack((cross_ds1_halo_res[:-1, i], cross_ds5_halo_res[1:, i])))[0][1]
    corrcoef_cross15auto1_dep[i] = np.corrcoef(np.vstack((cross_ds1_ds5_res[:, i], auto_ds1_res[:, i])))[0][1]
    corrcoef_cross15auto1_indep[i] = np.corrcoef(np.vstack((cross_ds1_ds5_res[:-1, i], auto_ds1_res[1:, i])))[0][1]
    corrcoef_cross15auto5_dep[i] = np.corrcoef(np.vstack((cross_ds1_ds5_res[:, i], auto_ds5_res[:, i])))[0][1]
    corrcoef_cross15auto5_indep[i] = np.corrcoef(np.vstack((cross_ds1_ds5_res[:-1, i], auto_ds5_res[1:, i])))[0][1]    
fig2, ax2 = plt.subplots(figsize=(6,4), dpi=500)
ax2.axhline(y=0, linewidth=0.5, linestyle='--', color='black')
ax2.plot(sep, corrcoef_auto15_dep, '-', label=r'$\Delta \epsilon_{11}, \Delta \epsilon_{55}$', color='tab:red')
ax2.plot(sep, corrcoef_auto15_indep, '--', color='tab:red')
ax2.plot(sep, corrcoef_crosshalo15_dep, '-', label=r'$\Delta \epsilon_{1h}, \Delta \epsilon_{5h}$', color='tab:orange')
ax2.plot(sep, corrcoef_crosshalo15_indep, '--', color='tab:orange')
ax2.plot(sep, corrcoef_cross15auto5_dep, '-', label=r'$\Delta \epsilon_{15}, \Delta \epsilon_{55}$', color='tab:green')
ax2.plot(sep, corrcoef_cross15auto5_indep, '--', color='tab:green')
ax2.plot(sep, corrcoef_cross15auto1_dep, '-', label=r'$\Delta \epsilon_{11}, \Delta \epsilon_{15}$', color='tab:blue')
ax2.plot(sep, corrcoef_cross15auto1_indep, '--', color='tab:blue')
ax2.legend(loc=1)
ax2.set_ylim(-1, 1)
ax2.set_xlabel(r'$s \ [h^{-1}Mpc]$')
ax2.set_ylabel(r'$\rho$')
ax2.set_title('Correlation Coefficients')



# function to compute the bias ratio estimate using relaxation method
def bias_ratio_estimate(init_ratio, corr_obj1, corr_obj2, dep_or_indep,
                        n_iterations):
    '''
    determines the bias ratio estimate using relaxation method (choosing
    an initial estimate, plug in then get another value, plug it back in, etc.)

    Parameters
    ----------
    init_ratio : float
        The initial bias ratio guess between corr_obj1 and corr_obj2.
    corr_obj1 : array
        The object 1 correlation values across all realizations
        (rows are separate realizations, columns are different separations).
    corr_obj2 : array
        The object 2 correlation values across all realizations.
    dep_or_indep : str
        Whether to compute using same ('dep') or different ('indep') realizations.
    n_iterations : int
        The number of iterations to apply for relaxation method.

    Returns
    -------
    ratios : list
        Distribution of bias ratios

    '''
    if dep_or_indep == 'dep': # from same realization
        for N in range(n_iterations):
           # print('Iteration {} bias ratio = '.format(N), init_ratio)
            dep_errs = corr_obj1[:-1]-init_ratio*corr_obj2[:-1]
            cov_mat = np.cov(m=dep_errs.T, bias=False)
            inv_cov_mat = np.linalg.inv(cov_mat)
            ratios = []
            for i in range(len(dep_errs)-1):
                corr_obj1_vec = np.array([corr_obj1[i]]).T
                corr_obj2_vec = np.array([corr_obj2[i]]).T
                corr_obj1_vec_trans = corr_obj1_vec.T
                corr_obj2_vec_trans = corr_obj2_vec.T
                ratio_now = ((np.matmul(np.matmul(corr_obj1_vec_trans, inv_cov_mat), corr_obj2_vec) + 
                              np.matmul(np.matmul(corr_obj2_vec_trans, inv_cov_mat), corr_obj1_vec))/
                             (2*np.matmul(np.matmul(corr_obj2_vec_trans, inv_cov_mat), corr_obj2_vec)))[0][0]
                ratios.append(ratio_now)
            init_ratio = np.mean(ratios)
        return ratios
        
    elif dep_or_indep == 'indep':
        for N in range(n_iterations):
          #  print('Iteration {} bias ratio = '.format(N), init_ratio)
            dep_errs = corr_obj1[:-1]-init_ratio*corr_obj2[1:]
            cov_mat = np.cov(m=dep_errs.T, bias=False)
            inv_cov_mat = np.linalg.inv(cov_mat)
            ratios = []
            for i in range(len(dep_errs)-1):
                corr_obj1_vec = np.array([corr_obj1[i]]).T
                corr_obj2_vec = np.array([corr_obj2[i+1]]).T
                corr_obj1_vec_trans = corr_obj1_vec.T
                corr_obj2_vec_trans = corr_obj2_vec.T
                ratio_now = ((np.matmul(np.matmul(corr_obj1_vec_trans, inv_cov_mat), corr_obj2_vec) + 
                              np.matmul(np.matmul(corr_obj2_vec_trans, inv_cov_mat), corr_obj1_vec))/
                             (2*np.matmul(np.matmul(corr_obj2_vec_trans, inv_cov_mat), corr_obj2_vec)))[0][0]
                ratios.append(ratio_now)
            init_ratio = np.mean(ratios)
        return ratios              
            


low_ind = 6
up_ind = 13
n_iter = 10
crosshalo_dep = bias_ratio_estimate(-1, cross_ds1_halo[:, low_ind:up_ind], cross_ds5_halo[:, low_ind:up_ind], 'dep', n_iter)
crosshalo_indep = bias_ratio_estimate(-1, cross_ds1_halo[:, low_ind:up_ind], cross_ds5_halo[:, low_ind:up_ind], 'indep', n_iter)
auto15_dep = -np.sqrt(np.array(bias_ratio_estimate(1, auto_ds1[:, low_ind:up_ind], auto_ds5[:, low_ind:up_ind], 'dep', n_iter)))
auto15_indep = -np.sqrt(np.array(bias_ratio_estimate(1, auto_ds1[:, low_ind:up_ind], auto_ds5[:, low_ind:up_ind], 'indep', n_iter)))                   
cross15auto5_dep = bias_ratio_estimate(-1, cross_ds1_ds5[:, low_ind:up_ind], auto_ds5[:, low_ind:up_ind], 'dep', n_iter)
cross15auto5_indep = bias_ratio_estimate(-1, cross_ds1_ds5[:, low_ind:up_ind], auto_ds5[:, low_ind:up_ind], 'indep', n_iter)
#auto1cross15_dep = bias_ratio_estimate(-1, cross_ds1_ds5[:, low_ind:up_ind], auto_ds1[:, low_ind:up_ind], 'dep', n_iter)
#auto1cross15_indep = bias_ratio_estimate(-1, cross_ds1_ds5[:, low_ind:up_ind], auto_ds1[:, low_ind:up_ind], 'indep', n_iter)
auto1cross15_dep = bias_ratio_estimate(-1, auto_ds1[:, low_ind:up_ind], cross_ds1_ds5[:, low_ind:up_ind], 'dep', n_iter)
auto1cross15_indep = bias_ratio_estimate(-1, auto_ds1[:, low_ind:up_ind], cross_ds1_ds5[:, low_ind:up_ind], 'indep', n_iter)
fig3, ax3 = plt.subplots(4, 2, dpi=500, figsize=(6, 8), sharex=True, sharey=True)
fig3.subplots_adjust(hspace=0.15, wspace=0.075)
ax3[0][0].set_title('Same Realization')
ax3[0][1].set_title('Different Realization')
ax3[0][0].hist(crosshalo_dep, np.arange(-1.4, -0.5, 0.02), density=False, color='red', alpha=1, label=r'$\epsilon_{1h}, \epsilon_{5h}$'); ax3[0][0].legend(loc=2); ax3[0][0].set_xticks([])
ax3[0][1].hist(crosshalo_indep, np.arange(-1.4, -0.5, 0.02), density=False, color='darkred', alpha=1, label=r'$\epsilon_{1h}, \epsilon_{5h}$'); ax3[0][1].legend(loc=2); ax3[0][1].set_xticks([])
ax3[1][0].hist(auto15_dep, np.arange(-1.4, -0.5, 0.02), density=False, color='orange', alpha=1, label=r'$\epsilon_{11}, \epsilon_{55}$'); ax3[1][0].legend(loc=2); ax3[1][0].set_xticks([])
ax3[1][1].hist(auto15_indep, np.arange(-1.4, -0.5, 0.02), density=False, color='darkorange', alpha=1, label=r'$\epsilon_{11}, \epsilon_{55}$'); ax3[1][1].legend(loc=2); ax3[1][1].set_xticks([])
ax3[2][0].hist(cross15auto5_dep, np.arange(-1.4, -0.5, 0.02), density=False, color='green', alpha=1, label=r'$\epsilon_{15}, \epsilon_{55}$'); ax3[2][0].legend(loc=2); ax3[2][0].set_xticks([])
ax3[2][1].hist(cross15auto5_indep, np.arange(-1.4, -0.5, 0.02), density=False, color='darkgreen', alpha=1, label=r'$\epsilon_{15}, \epsilon_{55}$'); ax3[2][1].legend(loc=2); ax3[2][1].set_xticks([])
ax3[3][0].hist(auto1cross15_dep, np.arange(-1.4, -0.5, 0.02), density=False, color='blue', alpha=1, label=r'$\epsilon_{11}, \epsilon_{15}$'); ax3[3][0].legend(loc=2)
ax3[3][1].hist(auto1cross15_indep, np.arange(-1.4, -0.5, 0.02), density=False, color='darkblue', alpha=1, label=r'$\epsilon_{11}, \epsilon_{15}$'); ax3[3][1].legend(loc=2)
ax3[3][0].set_xlabel(r'$b_1/b_5$')
ax3[3][1].set_xlabel(r'$b_1/b_5$')
ax3[3][0].set_xticks([-1.2, -1.0, -0.8, -0.6])
ax3[3][0].ticklabel_format(axis='both', scilimits=(0,0))
#ax3[0][0].set_yticks([0, 2, 4, 6])


def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

plot_vals = np.linspace(-1.4, -0.2, 1000)
#ax3[0].plot(plot_vals, gaussian(plot_vals, np.mean(crosshalo_dep), np.std(crosshalo_dep)), '-', color='tab:blue', linewidth=0.5)
#ax3[1].plot(plot_vals, gaussian(plot_vals, np.mean(crosshalo_indep), np.std(crosshalo_indep)), '-', color='tab:green', linewidth=0.5)

print(np.mean(crosshalo_dep), np.mean(crosshalo_indep))
print(np.mean(auto15_dep), np.mean(auto15_indep))
print(np.mean(cross15auto5_dep), np.mean(cross15auto5_indep))
print(np.mean(auto1cross15_dep), np.mean(auto1cross15_indep))
print('\n')
print(np.std(crosshalo_dep), np.std(crosshalo_indep))
print(np.std(auto15_dep), np.std(auto15_indep))
print(np.std(cross15auto5_dep), np.std(cross15auto5_indep))
print(np.std(auto1cross15_dep), np.std(auto1cross15_indep))
print('\n')
print(np.std(crosshalo_dep)/np.abs(np.mean(crosshalo_dep)))
print(np.std(crosshalo_indep)/np.abs(np.mean(crosshalo_indep)))
print(np.std(auto15_dep)/np.abs(np.mean(auto15_dep)))
print(np.std(auto15_indep)/np.abs(np.mean(auto15_indep)))
print(np.std(cross15auto5_dep)/np.abs(np.mean(cross15auto5_dep)))
print(np.std(cross15auto5_indep)/np.abs(np.mean(cross15auto5_indep)))
print(np.std(auto1cross15_dep)/np.abs(np.mean(auto1cross15_dep)))
print(np.std(auto1cross15_indep)/np.abs(np.mean(auto1cross15_indep)))
print('\n')
print(np.std(crosshalo_indep)/np.std(crosshalo_dep))
print(np.std(auto15_indep)/np.std(auto15_dep))
print(np.std(cross15auto5_indep)/np.std(cross15auto5_dep))
print(np.std(auto1cross15_indep)/np.std(auto1cross15_dep))




















        
        
        
        