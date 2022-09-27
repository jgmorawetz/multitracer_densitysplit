import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from itertools import combinations
import time

def dist_3d(x_1, y_1, z_1, x_2, y_2, z_2):
    return ((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)**0.5

def local_density(coords, tree, smooth_radius):
    num_neighbors = tree.query_radius(coords, smooth_radius, count_only=True)
    density = num_neighbors / (4*np.pi/3*smooth_radius**3)
    return density

def density_contrast(density, mean_density):
    return (density - mean_density)/mean_density



if __name__ == '__main__':
    t0 = time.time()
    path = '/home/jgmorawe/results/quantiles_zsplit_base_c000_ph000_z0.575_nden3.2e-04.npy'
    ds_quintiles = np.load(path)
    ds5_coords = ds_quintiles[4]
    tot = len(ds5_coords)
    vol_tot = 2000**3
    kdtree = KDTree(ds5_coords)
    
    sep_bins = np.arange(0, 700, 0.1)
    dens_cont_prod_sum = np.zeros(len(sep_bins)-1)
    num_pairs = np.zeros(len(sep_bins)-1)
    
    #boxside = 2000
    MIN = 800
    MAX = 1200
    # filters for small subset between 950 and 1050 in all spatial directions
    filter_ind = ((ds5_coords[:, 0] < MAX) & (ds5_coords[:, 0] > MIN) &
                  (ds5_coords[:, 1] < MAX) & (ds5_coords[:, 1] > MIN) & 
                  (ds5_coords[:, 2] < MAX) & (ds5_coords[:, 2] > MIN))
    ds5_coords = ds5_coords[filter_ind]
    local_dens = local_density(ds5_coords, kdtree, 15)
    dens_contrast = density_contrast(local_dens, tot/vol_tot)
    
    ds5_x = ds5_coords[:, 0]
    ds5_y = ds5_coords[:, 1]
    ds5_z = ds5_coords[:, 2]
    print('Finished retrieving data: ', time.time()-t0)
    
    t0 = time.time()
    for ind1, ind2 in combinations(np.arange(0, len(ds5_coords)), 2):
        x_1, x_2 = ds5_x[ind1], ds5_x[ind2]
        y_1, y_2 = ds5_y[ind1], ds5_y[ind2]
        z_1, z_2 = ds5_z[ind1], ds5_z[ind2]
        dist = dist_3d(x_1, y_1, z_1, x_2, y_2, z_2)
        dens_cont = dens_contrast[ind1]*dens_contrast[ind2]
        bin_ind = np.digitize(dist, sep_bins) - 1
        dens_cont_prod_sum[bin_ind] += dens_cont
        num_pairs[bin_ind] += 1
    
    sep_means = np.array([np.mean(sep_bins[i:i+2]) for i in range(len(sep_bins)-1)])
    dens_cont_prod_mean = dens_cont_prod_sum / num_pairs
    np.savetxt('sep_means.txt', sep_means)
    np.savetxt('dens_cont_prod_mean.txt', dens_cont_prod_mean)
    print('Finished algorithm: ', time.time()-t0)
    
    
        
        