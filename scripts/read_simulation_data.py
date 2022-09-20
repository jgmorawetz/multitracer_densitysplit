from astropy.io import fits
import numpy as np
from densitysplit.pipeline import DensitySplit
from pathlib import Path


def get_data_positions(data_fn, split='z', los='z'):
    """Return halo positions in real or redshift space
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

    # define simulation and algorithm parameters
    boxsize = 2000
    split = 'z'
    smooth_ds = 20
    nquantiles = 5
    los = 'z'
    cosmo = 0
    phase = 0
    redshift = 0.575
    cellsize = 5.0

    # run density split algorithm
    data_dir = '/home/epaillas/data/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.575'
    data_fn = Path(data_dir, 
        f'halos_base_c{cosmo:03}_ph{phase:03}_z{redshift:.3f}_nden3.2e-04.fits')

    data_positions = get_data_positions(data_fn, split=split, los=los)
    ndata = len(data_positions)

    ds = DensitySplit(data_positions, boxsize)

    sampling_positions = np.random.uniform(0, boxsize, (5 * ndata, 3))

    density = ds.get_density(smooth_radius=smooth_ds, cellsize=cellsize,
        sampling_positions=sampling_positions)

    quantiles = ds.get_quantiles(nquantiles=5)

    # save results
    output_dir = './'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_fn = Path(output_dir,
        f'quantiles_{split}split_base_c{cosmo:03}_ph{phase:03}_z{redshift:.3f}_nden3.2e-04.npy')

    np.save(output_fn, quantiles)
