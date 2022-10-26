
import time
import os
import pickle
import numpy as np
from pycorr import TwoPointCorrelationFunction


if __name__ == '__main__':
    
    t0 = time.time()
    sim_nums = np.arange(3000, 5000)
    sim_path = '/home/jgmorawe/results/quintiles/real'
    correlation_path = '/home/jgmorawe/results/correlation'
    
    for sim_num in sim_nums:
        
        if os.path.exists(os.path.join(sim_path, 'sim_{}_ds1.npy'.format(sim_num))):
            ds1 = np.load(os.path.join(sim_path, 'sim_{}_ds1.npy'.format(sim_num)))
            ds2 = np.load(os.path.join(sim_path, 'sim_{}_ds2.npy'.format(sim_num)))
            ds3 = np.load(os.path.join(sim_path, 'sim_{}_ds3.npy'.format(sim_num)))
            ds4 = np.load(os.path.join(sim_path, 'sim_{}_ds4.npy'.format(sim_num)))
            ds5 = np.load(os.path.join(sim_path, 'sim_{}_ds5.npy'.format(sim_num)))
            ds = np.vstack((ds1, ds2, ds3, ds4, ds5)) # all halo positions (from all densities)
            ds_groups = [ds1, ds2, ds3, ds4, ds5]
            density_labels = ['1', '2', '3', '4', '5']
            
            for i in range(len(ds_groups)):
                ds_group = ds_groups[i]
                density_label = density_labels[i]
                cross = TwoPointCorrelationFunction(mode='s', edges=np.linspace(0, 120, 25),
                                                    data_positions1=ds_group, data_positions2=ds,
                                                    position_type='pos', los='z', boxsize=500)
                pickle.dump(cross, open(os.path.join(correlation_path, 
                                                     'sim_{0}_cross{1}.pkl'.format(sim_num, density_label)), 'wb'))
            print('sim {} done'.format(sim_num))
        else:
            continue
    
    print('Program executed in {} seconds.'.format(time.time()-t0))