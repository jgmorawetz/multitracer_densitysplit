"""
This script makes a cross section plot of the density quintiles.
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from densitysplit.pipeline import DensitySplit



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
    
    # set the relevant parameters
    split = 'r'
    los = 'z'
    boxsize = 500
    smooth_ds = 20
    nquantiles = 5
    cellsize = 5
    sim_num = 3808
    cross_section_dim = 'z'
    cross_section_val = boxsize/2
    
    save_path = os.path.join('/home/jgmorawe/results/plot_data',
                             'cross_section_sim{0}_z={1}.png'.format(
                                 sim_num, int(cross_section_val)))
    
    data_fn = os.path.join(
        '/home/jgmorawe/projects/rrg-wperciva/AbacusSummit/small',
        'AbacusSummit_small_c000_ph{}/halos/z0.575'.format(sim_num),
        'halos_small_c000_ph{}_z0.575_nden3.2e-04.fits'.format(sim_num))
    
    halo_positions = get_data_positions(data_fn=data_fn, split=split, los=los)
    ds_object = DensitySplit(data_positions=halo_positions, boxsize=boxsize)
    
    grid_dim = 500
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                           figsize=(17, 5), dpi=500, 
                           gridspec_kw={'width_ratios': [1, 1.2, 1]})
    fig.subplots_adjust(wspace=0.1)
    
    if cross_section_dim == 'z':
        x_centres = np.linspace(0, boxsize, grid_dim)
        y_centres = np.linspace(0, boxsize, grid_dim)
        xy_grid = np.meshgrid(x_centres, y_centres)
        sampling_x = xy_grid[0].flatten()
        sampling_y = xy_grid[1].flatten()
        sampling_z = np.array([cross_section_val for i in range(grid_dim**2)])
        sampling_positions = np.vstack((sampling_x, sampling_y, sampling_z)).T
        density = ds_object.get_density(
            smooth_radius=smooth_ds, cellsize=cellsize,
            sampling_positions=sampling_positions)
        quantiles = ds_object.get_quantiles(nquantiles=nquantiles)
        density1 = ds_object.get_density(
            smooth_radius=smooth_ds, cellsize=cellsize, 
            sampling_positions=quantiles[0])
        density2 = ds_object.get_density(
            smooth_radius=smooth_ds, cellsize=cellsize, 
            sampling_positions=quantiles[1])
        density3 = ds_object.get_density(
            smooth_radius=smooth_ds, cellsize=cellsize, 
            sampling_positions=quantiles[2])
        density4 = ds_object.get_density(
            smooth_radius=smooth_ds, cellsize=cellsize, 
            sampling_positions=quantiles[3])
        density5 = ds_object.get_density(
            smooth_radius=smooth_ds, cellsize=cellsize, 
            sampling_positions=quantiles[4])
        avg_dens1 = np.mean(density1)
        avg_dens2 = np.mean(density2)
        avg_dens3 = np.mean(density3)
        avg_dens4 = np.mean(density4)
        avg_dens5 = np.mean(density5)
        avg_densities = [avg_dens1, avg_dens2, avg_dens3, avg_dens4, avg_dens5]
        halo_positions_slice = halo_positions[
            (halo_positions[:, 2] < cross_section_val + smooth_ds/2) &
            (halo_positions[:, 2] > cross_section_val - smooth_ds/2)]
        ax[0].plot(halo_positions_slice[:, 0], halo_positions_slice[:, 1], 'o', 
                   color='black', markersize=2)
        cmap='turbo'
        im=ax[1].scatter(x=sampling_x, y=sampling_y, s=2, c=density, cmap=cmap, 
                         vmin=-1, vmax=4, marker='s')
        fig.colorbar(im, ax=ax[1], pad=0.025)
        for i, mean_dens in zip(range(0, 5), avg_densities):
            size = len(quantiles[i][:,0])
            ax[2].scatter(x=quantiles[i][:, 0], y=quantiles[i][:, 1], s=2, 
                          c=np.full(size, mean_dens),
                          cmap=cmap, label='DS{}'.format(i+1), vmin=-1, vmax=4, 
                          marker='s')
        ax[0].set_xlim(0, boxsize); ax[0].set_ylim(0, boxsize)
        ax[1].set_xlim(0, boxsize); ax[1].set_ylim(0, boxsize)
        ax[2].set_xlim(0, boxsize); ax[2].set_ylim(0, boxsize)
        ax[0].set_ylabel('Y (Mpc/h)')
        ax[0].set_xlabel('X (Mpc/h)')
        ax[1].set_xlabel('X (Mpc/h)')
        ax[2].set_xlabel('X (Mpc/h)')
        ax[0].set_title(r'Halo Positions: Z $\in$ ({0},{1})'.format(
            int(cross_section_val - smooth_ds/2), 
            int(cross_section_val + smooth_ds/2)))
        ax[1].set_title('Smoothed Overdensity: Z={0}, $R_s$ = {1}'.format(
            int(cross_section_val), int(smooth_ds)))
        ax[2].set_title('Quintiles')
        ax[2].legend()
        fig.savefig(save_path)
                    
    else:
        print('Must plot for a z cross section.')