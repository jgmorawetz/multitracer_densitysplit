#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=power_spectra
#SBATCH --account=rrg-wperciva
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
python get_power_spectra.py