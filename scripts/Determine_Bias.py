
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import curve_fit
#from pycorr import TwoPointCorrelationFunction


def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x - mu)**2/(2*sigma**2))

if __name__ == '__main__':
    
    t0 = time.time()
    sim_nums = np.arange(3000, 5000)
    sim_path = '/home/jgmorawe/results/quintiles/real'
    correlation_path = '/home/jgmorawe/results/correlation'
    
    #num_sims = 0
    avgs2_1 = []
    avgs1_3 = []
    avgs1_4 = []
    avgs1_5 = []
    avgs2_3 = []
    avgs2_4 = []
    avgs2_5 = []
    avgs3_4 = []
    avgs3_5 = []
    avgs4_5 = []
    #ds1_sum = np.zeros(24)
    #ds2_sum = np.zeros(24)
    #ds3_sum = np.zeros(24)
    #ds4_sum = np.zeros(24)
    #ds5_sum = np.zeros(24)
    
    fig, ax = plt.subplots(5, 2, figsize=(5, 7.5), sharex=True, sharey=False, dpi=500)
    fig.subplots_adjust(wspace=0)
    
    for sim_num in sim_nums:
        if os.path.exists(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num))):
            corr_ds1 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).corr; #ds1_sum += corr_ds1
            corr_ds2 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross2.pkl'.format(sim_num)), 'rb')).corr; #ds2_sum += corr_ds2
            corr_ds3 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross3.pkl'.format(sim_num)), 'rb')).corr; #ds3_sum += corr_ds3
            corr_ds4 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross4.pkl'.format(sim_num)), 'rb')).corr; #ds4_sum += corr_ds4
            corr_ds5 = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross5.pkl'.format(sim_num)), 'rb')).corr; #ds5_sum += corr_ds5
            sep = pickle.load(open(os.path.join(correlation_path, 'sim_{}_cross1.pkl'.format(sim_num)), 'rb')).sep
            #num_sims += 1
            
            corr2_1 = np.mean(corr_ds2/corr_ds1); avgs2_1.append(corr2_1)
            corr1_3 = np.mean(corr_ds1/corr_ds3); avgs1_3.append(corr1_3) 
            corr1_4 = np.mean(corr_ds1/corr_ds4); avgs1_4.append(corr1_4)
            corr1_5 = np.mean(corr_ds1/corr_ds5); avgs1_5.append(corr1_5)
            corr2_3 = np.mean(corr_ds2/corr_ds3); avgs2_3.append(corr2_3)
            corr2_4 = np.mean(corr_ds2/corr_ds4); avgs2_4.append(corr2_4)
            corr2_5 = np.mean(corr_ds2/corr_ds5); avgs2_5.append(corr2_5)
            corr3_4 = np.mean(corr_ds3/corr_ds4); avgs3_4.append(corr3_4)
            corr3_5 = np.mean(corr_ds3/corr_ds5); avgs3_5.append(corr3_5)
            corr4_5 = np.mean(corr_ds4/corr_ds5); avgs4_5.append(corr4_5)
            
           # ax.plot(sep, sep**2*corr_ds1, '.', markersize=0.1, color='red')
           # ax.plot(sep, sep**2*corr_ds2, '.', markersize=0.1, color='orange')
           # ax.plot(sep, sep**2*corr_ds3, '.', markersize=0.1, color='green')
           # ax.plot(sep, sep**2*corr_ds4, '.', markersize=0.1, color='blue')
           # ax.plot(sep, sep**2*corr_ds5, '.', markersize=0.1, color='indigo')
            
        else:
            continue
    # filters ahead of time for 'bad cases' where the numbers are 'way off'
    
    avgs2_1 = list(filter(lambda x: abs(x) < 4, avgs2_1))
    avgs1_3 = list(filter(lambda x: abs(x) < 4, avgs1_3))
    avgs1_4 = list(filter(lambda x: abs(x) < 4, avgs1_4))
    avgs1_5 = list(filter(lambda x: abs(x) < 4, avgs1_5))
    avgs2_3 = list(filter(lambda x: abs(x) < 4, avgs2_3))
    avgs2_4 = list(filter(lambda x: abs(x) < 4, avgs2_4))
    avgs2_5 = list(filter(lambda x: abs(x) < 4, avgs2_5))
    avgs3_4 = list(filter(lambda x: abs(x) < 4, avgs3_4))
    avgs3_5 = list(filter(lambda x: abs(x) < 4, avgs3_5))
    avgs4_5 = list(filter(lambda x: abs(x) < 4, avgs4_5))
    
    
    mu21, sigma21 = np.mean(avgs2_1), np.std(avgs2_1); print(mu21)
    mu13, sigma13 = np.mean(avgs1_3), np.std(avgs1_3); print(mu13)
    mu14, sigma14 = np.mean(avgs1_4), np.std(avgs1_4); print(mu14)
    mu15, sigma15 = np.mean(avgs1_5), np.std(avgs1_5); print(mu15)
    mu23, sigma23 = np.mean(avgs2_3), np.std(avgs2_3); print(mu23)
    mu24, sigma24 = np.mean(avgs2_4), np.std(avgs2_4); print(mu24)
    mu25, sigma25 = np.mean(avgs2_5), np.std(avgs2_5); print(mu25)
    mu34, sigma34 = np.mean(avgs3_4), np.std(avgs3_4); print(mu34)
    mu35, sigma35 = np.mean(avgs3_5), np.std(avgs3_5); print(mu35)
    mu45, sigma45 = np.mean(avgs4_5), np.std(avgs4_5); print(mu45)
    
    
    ax[0][0].hist(avgs2_1, bins=np.linspace(-3, 3, 121), label=r'$b_2/b_1$', density=True); ax[0][0].legend()
    #ax[0][0].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu21, sigma21), 'r-')
    ax[0][0].axvline(mu21, linestyle='--', linewidth=1, color='red')
    ax[0][0].set_yticks([])
    
    ax[1][0].hist(avgs1_3, bins=np.linspace(-3, 3, 121), label=r'$b_1/b_3$', density=True); ax[1][0].legend()
    #ax[1][0].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu13, sigma13), 'r-')
    ax[1][0].axvline(mu13, linestyle='--', linewidth=1, color='red')
    ax[1][0].set_yticks([])
    
    ax[2][0].hist(avgs1_4, bins=np.linspace(-3, 3, 121), label=r'$b_1/b_4$', density=True); ax[2][0].legend()
    #ax[2][0].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu14, sigma14), 'r-')
    ax[2][0].axvline(mu14, linestyle='--', linewidth=1, color='red')
    ax[2][0].set_yticks([])
    
    ax[3][0].hist(avgs1_5, bins=np.linspace(-3, 3, 121), label=r'$b_1/b_5$', density=True); ax[3][0].legend()
   # ax[3][0].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu15, sigma15), 'r-')
    ax[3][0].axvline(mu15, linestyle='--', linewidth=1, color='red')
    ax[3][0].set_yticks([])
    
    ax[4][0].hist(avgs2_3, bins=np.linspace(-3, 3, 121), label=r'$b_2/b_3$', density=True); ax[4][0].legend()
   # ax[4][0].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu23, sigma23), 'r-')
    ax[4][0].axvline(mu23, linestyle='--', linewidth=1, color='red')
    ax[4][0].set_yticks([])
    
    ax[0][1].hist(avgs2_4, bins=np.linspace(-3, 3, 121), label=r'$b_2/b_4$', density=True); ax[0][1].legend()
   # ax[0][1].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu24, sigma24), 'r-')
    ax[0][1].axvline(mu24, linestyle='--', linewidth=1, color='red')
    ax[0][1].set_yticks([])
    
    ax[1][1].hist(avgs2_5, bins=np.linspace(-3, 3, 121), label=r'$b_2/b_5$', density=True); ax[1][1].legend()
   # ax[1][1].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu25, sigma25), 'r-')
    ax[1][1].axvline(mu25, linestyle='--', linewidth=1, color='red')
    ax[1][1].set_yticks([])
    
    ax[2][1].hist(avgs3_4, bins=np.linspace(-3, 3, 121), label=r'$b_3/b_4$', density=True); ax[2][1].legend()
    #ax[2][1].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu34, sigma34), 'r-')
    ax[2][1].axvline(mu34, linestyle='--', linewidth=1, color='red')
    ax[2][1].set_yticks([])
    
    ax[3][1].hist(avgs3_5, bins=np.linspace(-3, 3, 121), label=r'$b_3/b_5$', density=True); ax[3][1].legend()
    #ax[3][1].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu35, sigma35), 'r-')
    ax[3][1].axvline(mu35, linestyle='--', linewidth=1, color='red')
    ax[3][1].set_yticks([])
    
    ax[4][1].hist(avgs4_5, bins=np.linspace(-3, 3, 121), label=r'$b_4/b_5$', density=True); ax[4][1].legend()
   # ax[4][1].plot(np.linspace(-3, 3, 10001), gaussian(np.linspace(-3, 3, 10001), mu45, sigma45), 'r-')
    ax[4][1].axvline(mu45, linestyle='--', linewidth=1, color='red')
    ax[4][1].set_yticks([])
    
   # corr_ds1 = ds1_sum / num_sims
   # corr_ds2 = ds2_sum / num_sims
   # corr_ds3 = ds3_sum / num_sims
   # corr_ds4 = ds4_sum / num_sims
   # corr_ds5 = ds5_sum / num_sims
    
  #  ax.plot(sep, corr_ds1/corr_ds5, '--')
    #ax.plot(sep, sep**2*corr_ds1, '-', color='red')
    #ax.plot(sep, sep**2*corr_ds2, '-', color='orange')
    #ax.plot(sep, sep**2*corr_ds3, '-', color='green')
    #ax.plot(sep, sep**2*corr_ds4, '-', color='blue')
   # ax.plot(sep, sep**2*corr_ds5, '-', color='indigo')
    fig.savefig('/home/jgmorawe/results/plot_data/all_correlations.png')
    
    
    #for sim_num in sim_nums:
    #    
    #    if os.path.exists(os.path.join(sim_path, 'sim_{}_ds1.npy'.format(sim_num))):
    #        ds1 = np.load(os.path.join(sim_path, 'sim_{}_ds1.npy'.format(sim_num)))
     #       ds2 = np.load(os.path.join(sim_path, 'sim_{}_ds2.npy'.format(sim_num)))
     #       ds3 = np.load(os.path.join(sim_path, 'sim_{}_ds3.npy'.format(sim_num)))
     #       ds4 = np.load(os.path.join(sim_path, 'sim_{}_ds4.npy'.format(sim_num)))
     #       ds5 = np.load(os.path.join(sim_path, 'sim_{}_ds5.npy'.format(sim_num)))
     #       ds = np.vstack((ds1, ds2, ds3, ds4, ds5)) # all halo positions (from all densities)
     #       ds_groups = [ds1, ds2, ds3, ds4, ds5]
       #     density_labels = ['1', '2', '3', '4', '5']
            
       #     for i in range(len(ds_groups)):
       #         ds_group = ds_groups[i]
        #        density_label = density_labels[i]
       #         cross = TwoPointCorrelationFunction(mode='s', edges=np.linspace(0, 125, 126),
        #                                            data_positions1=ds_group, data_positions2=ds,
       #                                             position_type='pos', los='z', boxsize=500)
       #         pickle.dump(cross, open(os.path.join(correlation_path, 
       #                                              'sim_{0}_cross{1}.pkl'.format(sim_num, density_label)), 'wb'))
       #     print('sim {} done'.format(sim_num))
      #  else:
      #      continue
    #
   # print('Program executed in {} seconds.'.format(time.time()-t0))