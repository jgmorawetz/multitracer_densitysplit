"""
This script compares what the density contrast looks like if we the field is
smoothed according to the actual halo positions in each quintile versus being 
smoothed according to random positions (which are 'partitioned' into the 
quintiles of the actual halo field).
"""



import os
import numpy as np
from astropy.io import fits
from densitysplit.pipeline import DensitySplit
import matplotlib.pyplot as plt


def get_data_positions(data_fn, split='z', los='z'):
    '''
    Returns halo positions in real or redshift space.
    '''
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
    split = 'r'
    boxsize = 500
    smooth_ds = 20
    nquantiles = 5
    los = 'z'
    cellsize = 5
    
    for sim_num in range(3800, 3810):
        try:
            #sim_num = 3988 # doesn't matter which realization we use for now
            data_fn = '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small/AbacusSummit_small_c000_ph{0}/halos/z0.575/halos_small_c000_ph{1}_z0.575_nden3.2e-04.fits'.format(sim_num, sim_num)
            
            halo_positions = get_data_positions(data_fn=data_fn, split=split, los=los)
            ndata = len(halo_positions)
            
            random_x = np.random.uniform(0, boxsize, 500000)
            random_y = np.random.uniform(0, boxsize, 500000)
            random_z = np.array([250 for i in range(500000)])
            random_samples = np.vstack((random_x, random_y, random_z)).T
            
            ds = DensitySplit(halo_positions, boxsize)
            density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                                     sampling_positions=random_samples)
            quantiles = ds.get_quantiles(nquantiles=5)
            
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15,5),dpi=300)
            inds = ((halo_positions[:, 2] < 260) & (halo_positions[:, 2] > 240))
            ax[0].plot(halo_positions[:,0][inds], halo_positions[:,1][inds], 'o', markersize=1)
            ax[1].scatter(x=random_x, y=random_y, s=0.1, c=density)
            ax[2].plot(quantiles[0][:,0], quantiles[0][:,1], 'o', markersize=1, color='red')
            ax[2].plot(quantiles[1][:,0], quantiles[1][:,1], 'o', markersize=1, color='orange')
            ax[2].plot(quantiles[2][:,0], quantiles[2][:,1], 'o', markersize=1, color='green')
            ax[2].plot(quantiles[3][:,0], quantiles[3][:,1], 'o', markersize=1, color='blue')
            ax[2].plot(quantiles[4][:,0], quantiles[4][:,1], 'o', markersize=1, color='indigo')
            fig.savefig('/home/jgmorawe/results/cross_section/cross_sections_density_{}.png'.format(sim_num))
            
            """
            # first we split the 'actual' halo positions into the different density quintiles
            ds = DensitySplit(halo_positions, boxsize)
            density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                                          sampling_positions=halo_positions)
            quantiles_halo = ds.get_quantiles(nquantiles=5) # ACTUAL HALO POSITION QUINTILES
            
            
            # now we partition randoms into the density quintiles of the smoothed field of actual halo field
            random_positions = np.random.uniform(0, boxsize, (5*ndata, 3))
            ds = DensitySplit(halo_positions, boxsize)
            density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                                          sampling_positions=random_positions)
            quantiles_rand = ds.get_quantiles(nquantiles=5) # RANDOM POSITION QUINTILES
            
            
            # now that we have the quintiles of both the halos and randoms, we now
            # run the density field again, to compute the smoothed density contrast
            # for each case to see if they are similar (and plots several cross sections)
            ds1_halo = DensitySplit(quantiles_halo[0], boxsize)
            ds5_halo = DensitySplit(quantiles_halo[-1], boxsize)
            
            ds1_rand = DensitySplit(quantiles_rand[0], boxsize)
            ds5_rand = DensitySplit(quantiles_rand[-1], boxsize)
            
            # cross section positions to use
            random_x = np.random.uniform(0, boxsize, 300000)
            random_y = np.random.uniform(0, boxsize, 300000)
            random_z = np.array([250 for i in range(300000)])
            random_samples = np.vstack((random_x, random_y, random_z)).T
            
            ds1_halo_density = ds1_halo.get_density(smooth_radius=smooth_ds, cellsize=cellsize, sampling_positions=random_samples)
            ds5_halo_density = ds5_halo.get_density(smooth_radius=smooth_ds, cellsize=cellsize, sampling_positions=random_samples)
            ds1_rand_density = ds1_rand.get_density(smooth_radius=smooth_ds, cellsize=cellsize, sampling_positions=random_samples)
            ds5_rand_density = ds5_rand.get_density(smooth_radius=smooth_ds, cellsize=cellsize, sampling_positions=random_samples)
            
            fig, ax = plt.subplots(2, 3, dpi=300, sharex=True, sharey=True, figsize=(10,7))
            ax[0][0].scatter(x=random_samples[:,0], y=random_samples[:,1], s=0.1, c=ds1_halo_density)
            ax[0][1].scatter(x=random_samples[:,0], y=random_samples[:,1], s=0.1, c=ds5_halo_density)
            ax[0][1].scatter(x=quantiles_halo[0]
            ax[1][0].scatter(x=random_samples[:,0], y=random_samples[:,1], s=0.1, c=ds1_rand_density)
            ax[1][1].scatter(x=random_samples[:,0], y=random_samples[:,1], s=0.1, c=ds5_rand_density)
            fig.savefig('/home/jgmorawe/results/cross_section/cross_sections_density_{}.png'.format(sim_num))
            """
        except:
            continue
    