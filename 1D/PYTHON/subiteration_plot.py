#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab

# LOAD DATA FROM FILES

GPU = ["TITAN_V", "GTX_1080Ti"]
tpb = [32, 64, 128, 256] # 512]

# TIME VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for g in range(0, len(gpu)):
	for t in range(0, len(tpb)):
		numOverlap = range(0,tpb[0],2);
		numSubIterations = np.log(tpb[0]) / np.log(2)
		pylab.figure()
        dataPath = "../SUBITERATIONS_RESULTS/" + GPU[g] + "/N1024_TOL1_TPB" + str(tpb[t]) + ".txt"
		data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
		for i in range(0, numSubIterations):
			index = range(0,numOverlap) + i * numOverlap
			pylab.plot(range(0,tpb[t],2), data[index,2], linewidth=2, label = 'k = ' + str(data[index[0],1]))
		pylab.xlabel(r'Overlap', fontsize = 20)
		pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
		pylab.xticks(fontsize = 16)
		pylab.yticks(fontsize = 16)
		pylab.legend()
		pylab.grid()
		pylab.tight_layout()
        figurePath = "../FIGURES/subiterations_time_tpb" + str(tpb[t]) + "_" + GPU[g] + ".png")
		pylab.savefig(figurePath)
        

# BEST TIME VS SUBITERATION FOR VARIOUS TPB (= 32, 64, 128, 256)
for g in range(0, len(gpu)):
	for t in range(0, len(tpb)):
		numOverlap = range(0,tpb[0],2);
		numSubIterations = np.log(tpb[0]) / np.log(2)
        best_times = np.zeros(numSubIterations, 1)
        subiterations_values = np.zeros(numSubIterations, 1)
        dataPath = "../SUBITERATIONS_RESULTS/" + GPU[g] + "/N1024_TOL1_TPB" + str(tpb[t]) + ".txt"
		data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
		for i in range(0, numSubIterations):
			index = range(0,numOverlap) + i * numOverlap
            best_times[i] = np.min(data[index, 2])
            subiteration_values[i] = data[index[0], 1]
		pylab.plot(subiteration_values, best_times, linewidth=2, label = 'tpb = ' + tpb[t])
    pylab.xlabel(r'Subiterations', fontsize = 20)
	pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
	pylab.xticks(fontsize = 16)
	pylab.yticks(fontsize = 16)
	pylab.legend()
	pylab.grid()
	pylab.tight_layout()
	figurePath = "../FIGURES/subiterations_time_best" + "_" + GPU[g] + ".png")
	pylab.savefig(figurePath)
