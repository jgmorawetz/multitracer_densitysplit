# This code reads in a sample of the Abacus Summit datasets and makes a # power spectrum out of both the auto and cross correlationimport timeimport numpy as np; np.random.seed(0)import matplotlib.pyplot as pltfrom pypower import CatalogFFTPowerif __name__ == '__main__':    t0 = time.time()    path = '/home/jgmorawe/results/quantiles_zsplit_base_c000_ph000_z0.575_nden3.2e-04.npy'    ds_quintiles = np.load(path)    ds5_coords = ds_quintiles[4]  #  shuffle_ind = np.arange(0, len(ds1_coords))  #  np.random.shuffle(shuffle_ind)   # ds5_coords = ds5_coords[shuffle_ind]    #ds5_x = ds5_coords[:, 0]   # ds5_y = ds5_coords[:, 1]   # ds5_z = ds5_coords[:, 2]        MIN = 950    MAX = 1050    # filters for small subset between 950 and 1050 in all spatial directions    filter_ind = ((ds5_coords[:, 0] < MAX) & (ds5_coords[:, 0] > MIN) &                  (ds5_coords[:, 1] < MAX) & (ds5_coords[:, 1] > MIN) &                   (ds5_coords[:, 2] < MAX) & (ds5_coords[:, 2] > MIN))    ds5_coords = ds5_coords[filter_ind]  #  print(ds1_coords)   # print(np.shape(ds1_coords)) #   ds2_coords = ds_quintiles[1]  #  ds3_coords = ds_quintiles[2]   # ds4_coords = ds_quintiles[3]    #ds5_coords = ds_quintiles[4]        # for now tries only using a subsample to test run time   # N = 10000    #kedges = np.linspace(0., 0.2, 11)    result = CatalogFFTPower(        data_positions1=[ds5_coords[:, 0], ds5_coords[:, 1], ds5_coords[:, 2]], cellsize=5,                             boxsize=2000)    print(result)    print(result.poles.k)    print(result.poles.power)    print('Execution time: ', time.time()-t0)