"""
This script is used to read in AbacusSummit simulation data (a cube of
x,y,z positions in redshift and real space), and split the positions into
density quantiles.
"""


import os
import numpy as np
from astropy.io import fits
from densitysplit.pipeline import DensitySplit


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
    for split in ['z', 'r']: # split is whether redshift or real space
        boxsize = 2000 # length of simulation cube
        smooth_ds = 20 # smooth radius
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
        ndata = len(data_positions)
        
        # initiates density split object
        ds = DensitySplit(data_positions, boxsize)
        
        # generates random sampling positions and computes density
        sampling_positions = np.random.uniform(0, boxsize, (5 * ndata, 3))
        density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
                                 sampling_positions=sampling_positions)
        
        # retrieves the positions in each quantile
        quantiles = ds.get_quantiles(nquantiles=5)
        
        # saves results
        output_fn = os.path.join('/home/jgmorawe/results/quintiles',
            f'quantiles_{split}split_base_c000_ph000_z0.575_nden3.2e-04.npy')
        np.save(output_fn, quantiles)