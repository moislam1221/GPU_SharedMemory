#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab

# LOAD DATA FROM FILES
cpuData = pylab.loadtxt(open("../RESULTS/CPU_N1024_TOL1.txt"), delimiter=' ', usecols=(0,1,2,3))
gpuData_TITAN = pylab.loadtxt(open("../RESULTS/GPU_N1024_TOL1_TITAN_V.txt"), delimiter=' ', usecols=(0,1,2,3))
sharedData_TITAN = pylab.loadtxt(open("../RESULTS/SHARED_N1024_TOL1_TITAN_V.txt"), delimiter=' ', usecols=(0,1,2,3,4))
gpuData_GTX = pylab.loadtxt(open("../RESULTS/GPU_N1024_TOL1_GTX_1080Ti.txt"), delimiter=' ', usecols=(0,1,2,3))
sharedData_GTX = pylab.loadtxt(open("../RESULTS/SHARED_N1024_TOL1_GTX_1080Ti.txt"), delimiter=' ', usecols=(0,1,2,3,4))

# POSTPROCESS CPU DATA
N = cpuData[0,0]
cpuTime = np.mean(cpuData[:,2])
numTrials = len(cpuData[:,2])

# POSTPROCESS GPU DATA
tpb = [32, 64, 128, 256, 512]
gpuTime_TITAN = np.zeros(len(tpb))
gpuTime_GTX = np.zeros(len(tpb))
for i in range(len(tpb)):
    gpuTime_TITAN[i] = np.mean(gpuData_TITAN[i*numTrials:(i+1)*numTrials-1,3])
    gpuTime_GTX[i] = np.mean(gpuData_TITAN[i*numTrials:(i+1)*numTrials-1,3])

# POSTPROCESS SHARED DATA
sharedTime_TITAN = np.zeros(len(tpb))
sharedTime_GTX = np.zeros(len(tpb))
for i in range(len(tpb)):
    sharedTime_TITAN[i] = np.mean(sharedData_TITAN[i*numTrials:(i+1)*numTrials-1,3])
    sharedTime_GTX[i] = np.mean(sharedData_GTX[i*numTrials:(i+1)*numTrials-1,3])

# CPU -> GPU
cpuTogpu_TITAN = cpuTime / np.array(gpuTime_TITAN)
gpuToshared_TITAN = np.array(gpuTime_TITAN) / np.array(sharedTime_TITAN)
cpuToshared_TITAN = np.array(cpuTime_TITAN) / np.array(sharedTime_TITAN)
cpuTogpu_GTX = cpuTime / np.array(gpuTime_GTX)
gpuToshared_GTX = np.array(gpuTime_GTX) / np.array(sharedTime_GTX)
cpuToshared_GTX = np.array(cpuTime_GTX) / np.array(sharedTime_GTX)

# COMPARISON OF CPU-GPU-SHARED
pylab.figure()
pylab.semilogy(tpb, np.ones(len(tpb))*cpuTime, '--', linewidth=2,label = 'CPU')
pylab.semilogy(tpb, gpuTime_TITAN, '-o', linewidth=2, label = 'TITAN V (Global Memory)')
pylab.semilogy(tpb, sharedTime_TITAN, '-o', linewidth=2, label = 'TITAN V (Shared Memory)')
pylab.semilogy(tpb, gpuTime_GTX, '-o', linewidth=2, label = 'GTX (Global Memory)')
pylab.semilogy(tpb, sharedTime_GTX, '-o', linewidth=2, label = 'GTX (Shared Memory)')
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
pylab.plot(tpb, cpuTogpu_TITAN, '-o', linewidth=2, label = 'TITAN: CPU TO GPU')
pylab.plot(tpb, gpuToshared_TITAN, '-o', linewidth=2, label = 'TITAN: GPU TO SHARED APPROACH')
pylab.plot(tpb, cpuToshared_TITAN, '-o', linewidth=2, label = 'TITAN: CPU TO SHARED APPROACH')
pylab.plot(tpb, cpuTogpu_GTX, '-o', linewidth=2, label = 'GTX: CPU TO GPU')
pylab.plot(tpb, gpuToshared_GTX, '-o', linewidth=2, label = 'GTX: GPU TO SHARED APPROACH')
pylab.plot(tpb, cpuToshared_GTX, '-o', linewidth=2, label = 'GTX: CPU TO SHARED APPROACH')
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Achieved', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/speedup.png')
