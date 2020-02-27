#!/usr/bin/env python

import os
import subprocess
import matplotlib
import numpy as np
import pylab

# In this python script, define the inputs that the rest of the C++ scripts will use
# This python file provides an easy way to run and plot performance for different parameters
# Inputs are as follows:
# 1: N (the number of DOFs in the problem)
# 2: residual_flag (if 1, the metric for convergence is the residual. if 0, the metric for convergence is the solution error)
# 3: tolerance_reduction_flag (if 1, user will specify in input 4 how much to reduce tolerance by. if 0, user will specify the exact tolerance they want)
# 4: tol_value (value corresponding to tolerance or tolerance reduction - based on input 3 value)

# Define inputs to all of the functions
Nx = 128
Ny = 128
residual_flag = 1
tolerance_value = 100
tolerance_reduction_flag = 1

# 1 - Run the main function
os.system('nvcc main_2D_poisson_automated.cu -o main')
os.system('./main ' + str(Nx) + ' ' + str(Ny) + ' ' + str(residual_flag) + ' ' + str(tolerance_value) + ' ' + str(tolerance_reduction_flag))
os.system('rm main')

# 2 - Run the overlap function
#os.system('nvcc test_overlap_automated.cu -o overlap')
#os.system('./overlap ' + str(N) + ' ' + str(residual_flag) + ' ' + str(tolerance_value) + ' ' + str(tolerance_reduction_flag))
#os.system('rm overlap')

# 3 - Run the subiteration function
#os.system('nvcc test_subiterations_automated.cu -o subiterations')
#os.system('./subiterations ' + str(N) + ' ' + str(residual_flag) + ' ' + str(tolerance_value) + ' ' + str(tolerance_reduction_flag))
#os.system('rm subiterations')

# Plotting

# 0 - Create directories inside of the FIGURES folder to store images
#os.system('./make_directories_automated.py {} {} {} {}'.format(Nx, Ny, residual_flag, tolerance_value, tolerance_reduction_flag))

# 1 - Plot initial comparison
print("============================================================================================")
print("1 - PLOTTING INITIAL COMPARISON")
#os.system('./postprocessing_initial_automated.py {} {} {} {}'.format(Nx, Ny, residual_flag, tolerance_value, tolerance_reduction_flag))

# 2 - Plot overlap results
print("============================================================================================")
print("2 - PLOTTING OVERLAP RESULTS")
#os.system('./overlap_plot_automated.py {} {} {} {}'.format(Nx, Ny, residual_flag, tolerance_value, tolerance_reduction_flag))

# 3 - Plot subiteration results
print("============================================================================================")
print("3 - PLOTTING SUBITERATION RESULTS")
#os.system('./subiteration_plot_automated.py {} {} {} {}'.format(Nx, Ny, residual_flag, tolerance_value, tolerance_reduction_flag))

# 4 - Plot optimized comparison
print("============================================================================================")
print("4 - PLOTTING OPTIMIZED COMPARISON")
#os.system('./postprocessing_optimal_automated.py {} {} {} {}'.format(Nx, Ny, residual_flag, tolerance_value, tolerance_reduction_flag))
