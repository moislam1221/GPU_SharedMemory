#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab
import sys
sys.path.insert(0, 'Helper')
from createFileStrings import fileNameOverlapResults
from createFileStrings import figuresSubfolderName

# PARSE INPUTS
nDim = sys.argv[1]
residual_convergence_metric_flag = sys.argv[2]
tolerance_value = sys.argv[3]
tolerance_reduction_flag = sys.argv[4]

# CREATE THE FIGURES SUBFOLDER FOLDER
SUBFOLDER = figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)

# THREADS PER BLOCK VALUES
tpb = [32, 64, 128, 256, 512]

# CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
for t in range(len(tpb)):
    threadsPerBlock = tpb[t]
    DATA_FILE_NAME = fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(DATA_FILE_NAME), delimiter=' ', usecols=(0,1,2,3,4))
    pylab.loglog(range(0,tpb[t],2), data[:,1], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Number of Cycles', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS/overlap_cycles.png")

# TIME VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
for t in range(len(tpb)):
    threadsPerBlock = tpb[t]
    DATA_FILE_NAME = fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(DATA_FILE_NAME), delimiter=' ', usecols=(0,1,2,3,4))
    pylab.loglog(range(0,tpb[t],2), data[:,2], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS/overlap_time.png")

# TIME/CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
for t in range(len(tpb)):
    threadsPerBlock = tpb[t]
    DATA_FILE_NAME = fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(DATA_FILE_NAME), delimiter=' ', usecols=(0,1,2,3,4))
    pylab.loglog(range(0,tpb[t],2), data[:,2] / data[:,1], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Time/Cycle [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS/overlap_timepercycle.png")

# SPEEDUP VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
for t in range(len(tpb)):
    threadsPerBlock = tpb[t]
    DATA_FILE_NAME = fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(DATA_FILE_NAME), delimiter=' ', usecols=(0,1,2,3,4))
    pylab.semilogx(range(0,tpb[t],2), data[0,2] / data[:,2], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Speedup (Relative to No Overlap)', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout() 
pylab.savefig("FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS/overlap_speedup.png")

# TIME VS OPERATIONAL THREADS FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
# pylab.figure()
# for t in range(len(tpb)):
#    threadsPerBlock = tpb[t]
#    DATA_FILE_NAME = fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
#    data = pylab.loadtxt(open(DATA_FILE_NAME), delimiter=' ', usecols=(0,1,2,3,4))
#    overlap = np.arange(0,tpb[t],2)
#    operational_threads = ((nDim - overlap) / (tpb[t] - overlap)) * tpb[t]
#    pylab.semilogy(operational_threads, data[:,2] / data[:,1], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
#pylab.xlabel(r'Number of Points to Update', fontsize = 20)
#pylab.ylabel(r'Time/Cycle [ms]', fontsize = 20)
#pylab.xticks(fontsize = 16)
#pylab.yticks(fontsize = 16)
#pylab.legend()
#pylab.grid()
#pylab.tight_layout()
#pylab.savefig("FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS/overlap_operationalpoints.png")

# PRINT BEST TIMES AND OVERLAP
best_times = np.zeros(len(tpb))
best_overlap = np.zeros(len(tpb))
for t in range(len(tpb)):
    threadsPerBlock = tpb[t]
    DATA_FILE_NAME = fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(DATA_FILE_NAME), delimiter=' ', usecols=(0,1,2,3,4))
    best_times[t] = np.min(data[:,2])
    best_index = np.where(data[:,2] == best_times[t])
    best_overlap[t] = data[best_index, 0]
print("The threads per block are: " + str(threadsPerBlock))
print("The best times achieved (SUBITERATIONS = TPB/2 fixed) per block are: " + str(best_times))
print("The corresponding overlap values are: " + str(best_overlap))
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS/overlap_time.png")
