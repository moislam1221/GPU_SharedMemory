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
from createFileStrings import figuresSubfolderName

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

# CPU -> GPU
cpuTogpu = cpuTime / np.array(gpuTime)
gpuToshared = min(np.array(gpuTime)) / np.array(sharedTime)
cpuToshared = np.array(cpuTime) / np.array(sharedTime)

# COMPARISON OF CPU-GPU-SHARED
pylab.figure()
pylab.loglog(tpb, np.ones(len(tpb))*cpuTime, '--', linewidth=2,label = 'CPU')
pylab.loglog(tpb, gpuTime, '-o', linewidth=2, label = 'GPU')
pylab.loglog(tpb, sharedTime, '-o', linewidth=2, label = 'Shared')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/INITIAL/cpu_gpu_shared.png")

# SPEEDUP BETWEEN CPU-SHARED
pylab.figure()
pylab.semilogx(tpb, cpuToshared, '-o', linewidth=2, label = 'CPU TO SHARED APPROACH')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Achieved', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/INITIAL/speedup_cpu_comparison.png")

# SPEEDUP BETWEEN GPU-SHARED
pylab.figure()
bestgpuToshared = np.array(np.ones(len(tpb)) * min(gpuTime)) / np.array(sharedTime)
pylab.semilogx(tpb, bestgpuToshared, '-o', linewidth=2, label = 'GPU TO SHARED APPROACH')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Achieved', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig("FIGURES/" + SUBFOLDER + "/INITIAL/speedup_gpu_comparison.png")

# PRINT OUT SOME IMPORTANT THINGS
print("INITIAL COMPARISON OF CPU/GPU/SHARED APPROACHES WITHOUT OPTIMIZATIONS")
print("The threads per block are: " + str(tpb)) 
print("The CPU times are: " + str(cpuTime))
print("The GPU times are: " + str(gpuTime))
print("The Shared times are: " + str(sharedTime))
print("The CPU to Shared speedups are: " + str(cpuToshared))
print("The GPU (best) to Shared speedups are: " + str(bestgpuToshared))
