#!/usr/bin/env python

import os
import subprocess
import matplotlib
import numpy as np
import pylab

# Define number of runs
trials = 20

# Perform commands to run test
os.system('nvcc ../main_1D_poisson.cu -o ../main')
for i in range(trials):
    print('Trial ' + str(i+1) + '/' + str(trials) + ' IN PROGRESS!!')
    os.system('.././main')
    print('Trial ' + str(i+1) + '/' + str(trials) + ' IS DONE!!')
    
