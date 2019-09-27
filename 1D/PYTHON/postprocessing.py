#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab

# LOAD DATA FROM FILES
cpuData = pylab.loadtxt(open("../RESULTS/CPU_N1024_TOL1.txt"), delimiter=' ', usecols=(0,1,2,3))
gpuData = pylab.loadtxt(open("../RESULTS/GPU_N1024_TOL1.txt"), delimiter=' ', usecols=(0,1,2,3,4))
sharedData = pylab.loadtxt(open("../RESULTS/SHARED_N1024_TOL1.txt"), delimiter=' ', usecols=(0,1,2,3,4))

# POSTPROCESS CPU DATA
N = cpuData[0,0]
cpuTime = np.mean(cpuData[:,2])
numTrials = len(cpuData[:,2])

# POSTPROCESS GPU DATA
tpb = [32, 64, 128, 256, 512]
gpuTime = np.zeros(len(tpb))
for i in range(len(tpb)):
    gpuTime[i] = np.mean(gpuData[i*numTrials:(i+1)*numTrials-1,3])

# POSTPROCESS GPU DATA
sharedTime = np.zeros(len(tpb))
for i in range(len(tpb)):
    sharedTime[i] = np.mean(sharedData[i*numTrials:(i+1)*numTrials-1,3])

# CPU -> GPU
cpuTogpu = cpuTime / np.array(gpuTime)
gpuToshared = np.array(gpuTime) / np.array(sharedTime)
cpuToshared = np.array(cpuTime) / np.array(sharedTime)

# COMPARISON OF CPU-GPU-SHARED
pylab.figure()
pylab.semilogy(tpb, np.ones(len(tpb))*cpuTime, '--', linewidth=2,label = 'CPU')
pylab.semilogy(tpb, gpuTime, '-o', linewidth=2, label = 'GPU')
pylab.semilogy(tpb, sharedTime, '-o', linewidth=2, label = 'Shared')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/cpu_gpu_shared.png')

# SPEEDUP BETWEEN CPU-GPU-SHARED
pylab.figure()
pylab.plot(tpb, cpuTogpu, '-o', linewidth=2, label = 'CPU TO GPU')
pylab.plot(tpb, gpuToshared, '-o', linewidth=2, label = 'GPU TO SHARED APPROACH')
pylab.plot(tpb, cpuToshared, '-o', linewidth=2, label = 'GPU TO SHARED APPROACH')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Achieved', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/speedup.png')
