#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab

# LOAD DATA FROM FILES
tpb32Data = pylab.loadtxt(open("../SUBITERATIONS_RESULTS/N1024_TOL1_TPB32.txt"), delimiter=' ', usecols=(0,1,2,3))
#tpb64Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB64.txt"), delimiter=' ', usecols=(0,1,2,3))
#tpb128Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB128.txt"), delimiter=' ', usecols=(0,1,2,3))
#tpb256Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB256.txt"), delimiter=' ', usecols=(0,1,2,3))
#tpb512Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB512.txt"), delimiter=' ', usecols=(0,1,2,3))

tpb = [32, 64, 128, 256, 512]

# TIME VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
numOverlap = range(0,tpb[0],2);
numSubIterations = np.log(tpb[0]) / np.log(2)
for i in range(0, numSubIterations):
	index = range(0,numOverlap) + i * numOverlap
	pylab.plot(range(0,tpb[0],2), tpb32Data[index,2], linewidth=2, label = 'hello')
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/subiterations_time.png')
