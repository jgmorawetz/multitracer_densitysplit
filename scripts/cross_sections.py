
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
    boxsize = 2000 # length of simulation cube
    smooth_ds = 10 # smooth radius
    nquantiles = 5 # number of percentiles to use
    los = 'z' # line of sight direction
    cellsize = 5 # size of the cells within the cube
    
    # run density split
    data_dir = os.path.join('/home/jgmorawe/projects/rrg-wperciva/',
                            'AbacusSummit/AbacusSummit_base_c000_ph000/',
                            'halos/z0.575')
    # uses cosmo = 0, phase = 0, redshift = 0.575 (may need to adjust)
    data_fn = os.path.join(data_dir,
                'halos_base_c000_ph000_z0.575_nden3.2e-04.fits')
    
    # retrieves x,y,z data positions and counts number of objects
    data_positions = get_data_positions(data_fn=data_fn, split=split, 
                                        los=los)
    data_positions_within = data_positions[(750 < data_positions[:,0]) &
                                           (1250 > data_positions[:,0]) &
                                           (750 < data_positions[:, 1]) &
                                           (1250 > data_positions[:, 1]) &
                                           (975 < data_positions[:, 2]) &
                                           (1025 > data_positions[:, 2])]
    ndata = len(data_positions)
    
    # initiates density split object
    ds = DensitySplit(data_positions, boxsize)
    
    # generates random points in a particular cross section to plot
    np.random.seed(0)
    random_x = np.random.uniform(750, 1250, 1000000)
    random_y = np.random.uniform(750, 1250, 1000000)
    random_z = np.random.uniform(999, 1001, 1000000)
    sampling_positions = np.c_[random_x, random_y, random_z]
    density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                             sampling_positions=sampling_positions)
    
    # retrieves the positions in each quantile
    quantiles = ds.get_quantiles(nquantiles=5)
    
    fig,ax=plt.subplots(dpi=300)
    ax.plot(quantiles[0][:, 0], quantiles[0][:, 1], 'o', color='red', markersize=0.5,label='DS1')
    ax.plot(quantiles[1][:, 0], quantiles[1][:, 1], 'o', color='orange', markersize=0.5, label='DS2')
    ax.plot(quantiles[2][:, 0], quantiles[2][:, 1], 'o', color='green', markersize=0.5,label='DS3')
    ax.plot(quantiles[3][:, 0], quantiles[3][:, 1], 'o', color='blue', markersize=0.5,label='DS4')
    ax.plot(quantiles[4][:, 0], quantiles[4][:, 1], 'o', color='darkviolet', markersize=0.5,label='DS5')
    ax.legend()
   # ax.plot(data_positions_within[:, 0], data_positions_within[:, 1], 'o', markersize=1.5,color='black')
    fig.savefig(os.path.join('/home/jgmorawe/results/cross_section',
                             'sample.png'))
    
    fig2, ax2=plt.subplots(dpi=300)
    ax2.plot(data_positions_within[:, 0], data_positions_within[:, 1], 'o', markersize=0.5,color='black')
    
    
    # saves results
    fig2.savefig(os.path.join('/home/jgmorawe/results/cross_section',
                              'sample_gal.png'))
    # output_fn = os.path.join('/home/jgmorawe/results/positions',
   #     'random_positions.npy')
   # np.save(output_fn, quantiles)
   # np.save(os.path.join('/home/jgmorawe/results/positions',
    #                     'galaxy_positions.npy'), data_positions_within)