""" 
This script reads in the generated correlation data for real and redshift
space positions for the AbacusSummit data, and creates plots.
"""

import os
import pickle
import matplotlib.pyplot as plt

        
if __name__ == '__main__':
    
    correlation_path = '/home/jgmorawe/results/correlation'
    
    # makes plot for autocorrelations first
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, dpi=300,
                           figsize=(7, 7))
    fig.subplots_adjust(hspace=0, wspace=0)
    colors = ['red', 'orange', 'green', 'blue', 'indigo']
    for i in range(1, 6):
        result_z = pickle.load(
                    open(os.path.join(
                        correlation_path, 'Auto_z_{}.pkl'.format(i)), 'rb'))
        result_r = pickle.load(
                    open(os.path.join(
                        correlation_path, 'Auto_r_{}.pkl'.format(i)), 'rb'))
        s_z, multipoles_z = result_z(ells=(0, 2), return_sep=True)
        s_r, multipoles_r = result_r(ells=(0, 2), return_sep=True)
        ax[0][0].plot(s_z, s_z**2*multipoles_z[0], '-', color=colors[i-1],
                      label='DS{}'.format(i))
        ax[1][0].plot(s_z, s_z**2*multipoles_z[1], '-', color=colors[i-1])
        ax[0][1].plot(s_r, s_r**2*multipoles_r[0], '-', color=colors[i-1])
        ax[1][1].plot(s_r, s_r**2*multipoles_r[1], '-', color=colors[i-1])
    ax[0][0].legend()
    ax[0][0].set_title('z-split')
    ax[0][1].set_title('r-split')
    ax[1][0].set_xlabel(r'$s [h^{-1} Mpc]$')
    ax[1][1].set_xlabel(r'$s [h^{-1} Mpc]$')
    ax[0][0].set_ylabel(r'$s^2 \xi_0 (s) [h^{-2} Mpc^2]$')
    ax[1][0].set_ylabel(r'$s^2 \xi_2 (s) [h^{-2} Mpc^2]$')
    fig.savefig(os.path.join(correlation_path, 'Auto_Plot.png'))
    
    # makes plot for cross correlations next
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, dpi=300,
                           figsize=(7, 7))
    fig.subplots_adjust(hspace=0, wspace=0)
    colors=['red', 'orange', 'green', 'blue', 'indigo']
    for i in range(1, 6):
        result_z = pickle.load(
                    open(os.path.join(
                        correlation_path, 'Cross_z_{}.pkl'.format(i)), 'rb'))
        result_r = pickle.load(
                    open(os.path.join(
                        correlation_path, 'Cross_r_{}.pkl'.format(i)), 'rb'))
        s_z, multipoles_z = result_z(ells=(0, 2), return_sep=True)
        s_r, multipoles_r = result_r(ells=(0, 2), return_sep=True)
        ax[0][0].plot(s_z, s_z**2*multipoles_z[0], '-', color=colors[i-1],
                      label='DS{}'.format(i))
        ax[1][0].plot(s_z, s_z**2*multipoles_z[1], '-', color=colors[i-1])
        ax[0][1].plot(s_r, s_r**2*multipoles_r[0], '-', color=colors[i-1])
        ax[1][1].plot(s_r, s_r**2*multipoles_r[1], '-', color=colors[i-1])
    ax[0][0].legend()
    ax[0][0].set_title('z-split')
    ax[0][1].set_title('r-split')
    ax[1][0].set_xlabel(r'$s [h^{-1} Mpc]$')
    ax[1][1].set_xlabel(r'$s [h^{-1} Mpc]$')
    ax[0][0].set_ylabel(r'$s^2 \xi_0 (s) [h^{-2} Mpc^2]$')
    ax[1][0].set_ylabel(r'$s^2 \xi_2 (s) [h^{-2} Mpc^2]$')
    fig.savefig(os.path.join(correlation_path, 'Cross_Plot.png'))