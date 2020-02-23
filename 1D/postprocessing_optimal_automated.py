#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab
import sys
sys.path.insert(0, 'Helper')
from createFileStrings import fileNameResults
from createFileStrings import fileNameSubiterationResults
from createFileStrings import figuresSubfolderName

# PARSE INPUTS
nDim = sys.argv[1]
residual_convergence_metric_flag = sys.argv[2]
tolerance_value = sys.argv[3]
tolerance_reduction_flag = sys.argv[4]

# CREATE FILE NAMES FROM WHICH TO LOAD
CPUDATA_FILE_NAME = fileNameResults("CPU", nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
GPUDATA_FILE_NAME = fileNameResults("GPU", nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
SHAREDDATA_FILE_NAME = fileNameResults("SHARED", nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)

# LOAD DATA FROM FILES
cpuData = pylab.loadtxt(open(CPUDATA_FILE_NAME), delimiter=' ', usecols=(2))
gpuData = pylab.loadtxt(open(GPUDATA_FILE_NAME), delimiter=' ', usecols=(3))
sharedData = pylab.loadtxt(open(SHAREDDATA_FILE_NAME), delimiter=' ', usecols=(3))

# CREATE THE FIGURES SUBFOLDER FOLDERFIGURES_SUBFOLDER = figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
SUBFOLDER = figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)

# THREADS PER BLOCK
tpb = [32, 64, 128, 256, 512, 1024]

# POSTPROCESS CPU DATA
cpuTime = cpuData

# POSTPROCESS GPU DATA
gpuTime = np.zeros(len(tpb))
for i in range(len(tpb)):
    gpuTime[i] = gpuData[i]

# POSTPROCESS SHARED DATA
sharedTime = np.zeros(len(tpb))
for i in range(len(tpb)):
    sharedTime[i] = sharedData[i]

# CREATE NECESSARY CONTAINERS
best_times = np.zeros(len(tpb))
best_overlap = np.zeros(len(tpb))
best_subiteration = np.zeros(len(tpb))

for t in range(len(tpb)):
    threadsPerBlock = tpb[t]
    dataPath = fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
    data = pylab.loadtxt(open(dataPath), delimiter=' ', usecols=(0,1,2,3))
    best_times[t] = np.min(data[:,3])
    best_index = np.where(data[:,3] == best_times[t]) 
    best_overlap[t] = data[best_index, 0]
    best_subiteration[t] = data[best_index, 1]

# CPU -> GPU
cpuTogpu = cpuTime / np.array(gpuTime)
gpuToshared = min(np.array(gpuTime)) / np.array(sharedTime)
cpuToshared = np.array(cpuTime) / np.array(sharedTime)
gpuTosharedOptimized = min(np.array(gpuTime)) / np.array(best_times)
cpuTosharedOptimized = np.array(cpuTime) / np.array(best_times)

# COMPARISON OF CPU-GPU-SHARED
pylab.figure()
pylab.semilogy(tpb, np.ones(len(tpb))*cpuTime, '--', linewidth=2,label = 'CPU')
pylab.loglog(tpb, gpuTime, '-o', linewidth=2, label = 'GPU')
pylab.loglog(tpb, sharedTime, '--o', linewidth=2, label = 'Shared')
pylab.loglog(tpb, best_times, '-o', linewidth=2, label = 'Shared (Optimized)')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OPTIMIZED/cpu_gpu_shared_optimized.png")

# SPEEDUP BETWEEN CPU-SHARED
pylab.figure()
pylab.semilogx(tpb, cpuToshared, '--o', linewidth=2, label = 'CPU to Shared')
pylab.semilogx(tpb, cpuTosharedOptimized, '-o', linewidth=2, label = 'CPU to Shared (Optimized)')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Achieved', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OPTIMIZED/speedup_cpu_comparison_optimized.png")

# SPEEDUP BETWEEN GPU-SHARED
pylab.figure()
pylab.semilogx(tpb, gpuToshared, '--o', linewidth=2, label = 'GPU to Shared')
pylab.semilogx(tpb, gpuTosharedOptimized, '-o', linewidth=2, label = 'GPU to Shared (Optimized)')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Achieved', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/OPTIMIZED/speedup_gpu_comparison_optimized.png")

print("FINAL COMPARISON OF CPU/GPU/SHARED APPROACHED WITH OPTIMIZATIONS")
print("The threads per block are " + str(tpb))
print("The CPU times are: " + str(cpuTime))
print("The GPU times are: " + str(gpuTime))
print("The Shared times are: " + str(sharedTime))
print("The best times are : " + str(best_times))
print("The respective optimal overlap values are: " + str(best_overlap))
print("The respective optimal subiterations are: " + str(best_subiteration))
print("The GPU Global to Shared approach speedups (original) are: " + str(gpuToshared))
print("The GPU Global to Shared approach speedups (optimized) are: " + str(gpuTosharedOptimized))
