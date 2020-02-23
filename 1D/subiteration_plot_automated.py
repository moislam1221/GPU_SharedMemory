#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab
import sys
sys.path.insert(0, 'Helper')
from createFileStrings import figuresSubfolderName
from createFileStrings import fileNameSubiterationResults

# PARSE INPUTS
nDim = sys.argv[1]
residual_convergence_metric_flag = sys.argv[2]
tolerance_value = sys.argv[3]
tolerance_reduction_flag = sys.argv[4]

# CREATE THE FIGURES SUBFOLDER FOLDER
SUBFOLDER = figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)

# THREADS PER BLOCK VALUES
tpb = [32, 64, 128, 256, 512]

# TIME VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for t in range(0, len(tpb)):
    numOverlap = len(range(0,tpb[t]/2+2,2))
    numSubIterations = 5
    # numSubIterations = int(np.log(tpb[t]) / np.log(2) + 2)
    threadsPerBlock = tpb[t]
    pylab.figure()
    dataPath = fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
    for i in range(0, numSubIterations):
        index = np.arange(0,numOverlap) + i * numOverlap
        pylab.semilogy(range(0,tpb[t]/2+2,2), data[index,3], '.-', linewidth=2, label = 'k = ' + str(data[index[0],1]))
	pylab.xlabel(r'Overlap', fontsize = 20)
	pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
	pylab.xticks(fontsize = 16)
	pylab.yticks(fontsize = 16)
	pylab.legend()
	pylab.grid()
	pylab.tight_layout()
	figurePath = "FIGURES/" + SUBFOLDER + "/SUBITERATION_RESULTS/subiterations_time_tpb" + str(tpb[t]) + ".png"
	pylab.savefig(figurePath)
        
# CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for t in range(0, len(tpb)):
    numOverlap = len(range(0,tpb[t]/2+2,2));
    numSubIterations = 5
    # numSubIterations = int(np.log(tpb[t]) / np.log(2) + 2)
    threadsPerBlock = tpb[t]
    pylab.figure()
    dataPath = fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
    for i in range(0, numSubIterations):
        index = np.arange(0,numOverlap) + i * numOverlap
        pylab.semilogy(range(0,tpb[t]/2+2,2), data[index,2], '.-', linewidth=2, label = 'k = ' + str(data[index[0],1]))
	pylab.xlabel(r'Overlap', fontsize = 20)
	pylab.ylabel(r'Cycles', fontsize = 20)
	pylab.xticks(fontsize = 16)
	pylab.yticks(fontsize = 16)
	pylab.legend()
	pylab.grid()
	pylab.tight_layout()
	figurePath = "FIGURES/" + SUBFOLDER + "/SUBITERATION_RESULTS/subiterations_cycles_tpb" + str(tpb[t]) + ".png"
    pylab.savefig(figurePath)

# TIME/CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for t in range(0, len(tpb)):
    numOverlap = len(range(0,tpb[t]/2+2,2));
    numSubIterations = 5
    # numSubIterations = int(np.log(tpb[t]) / np.log(2) + 2)
    threadsPerBlock = tpb[t]    
    pylab.figure()
    dataPath = fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
    for i in range(0, numSubIterations):
        index = np.arange(0,numOverlap) + i * numOverlap
        pylab.semilogy(range(0,tpb[t]/2+2,2), data[index,3] / data[index,2], '.-', linewidth=2, label = 'k = ' + str(data[index[0],1]))
	pylab.xlabel(r'Overlap', fontsize = 20)
	pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
	pylab.xticks(fontsize = 16)
	pylab.yticks(fontsize = 16)
	pylab.legend()
	pylab.grid()
	pylab.tight_layout()
	figurePath = "FIGURES/" + SUBFOLDER + "/SUBITERATION_RESULTS/subiterations_timepercycle_tpb" + str(tpb[t]) + ".png"
    pylab.savefig(figurePath)

# BEST TIME VS SUBITERATION FOR VARIOUS TPB (= 32, 64, 128, 256)
for t in range(0, len(tpb)):
    numOverlap = len(range(0,tpb[t]/2+2,2))
    numSubIterations = 5
    # numSubIterations = int(np.log(tpb[t]) / np.log(2) + 2)
    threadsPerBlock = tpb[t]    
    pylab.figure()
    best_times = np.zeros(numSubIterations)
    subiteration_values = np.zeros(numSubIterations)
    dataPath = fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
    for i in range(0, numSubIterations):
        index = np.arange(0,numOverlap) + i * numOverlap
        best_times[i] = np.min(data[index, 3])
        subiteration_values[i] = data[index[0], 1]
    pylab.semilogy(subiteration_values, best_times, 'o-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
    pylab.xlabel(r'Subiterations', fontsize = 20)
    pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.tight_layout()
    figurePath = "FIGURES/" + SUBFOLDER + "/SUBITERATION_RESULTS/subiterations_time_best_tpb" + str(tpb[t]) + ".png"
    pylab.savefig(figurePath)

# BEST OVERLAP VS SUBITERATION FOR VARIOUS TPB (= 32, 64, 128, 256)
for t in range(0, len(tpb)):
    numOverlap = len(range(0,tpb[t]/2+2,2));
    numSubIterations = 5
    # numSubIterations = int(np.log(tpb[t]) / np.log(2) + 2)
    threadsPerBlock = tpb[t]
    best_overlap = np.zeros(numSubIterations)
    best_index = np.zeros(numSubIterations)
    best_times = np.zeros(numSubIterations)
    subiteration_values = np.zeros(numSubIterations)
    dataPath = fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
    for i in range(0, numSubIterations):
		index = np.arange(0,numOverlap) + i * numOverlap
		best_times[i] = np.min(data[index, 3])
		subiteration_values[i] = data[index[0], 1]
		best_index[i] = np.argmin(data[index, 3])
		overlap_array = data[index, 0]
		best_overlap[i] = overlap_array[int(best_index[i])]
    print("THREADS PER BLOCK = " + str(threadsPerBlock) + ":")
    print("Subiterations explored are = " + str(subiteration_values))
    print("The best times achieved (for a tpb, subiteration value) are " + str(best_times))
    print("The corresponding overlap values are " + str(best_overlap) + " out of max possible overlap of " + str(threadsPerBlock-2))
    pylab.semilogy(subiteration_values, best_overlap, 'o-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
    pylab.xlabel(r'Subiterations', fontsize = 20)
    pylab.ylabel(r'Optimal Overlap', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.tight_layout() 
    figurePath = "FIGURES/" + SUBFOLDER + "/SUBITERATION_RESULTS/subiterations_overlap_best_tpb" + str(tpb[t]) + ".png"
    pylab.savefig(figurePath)	
