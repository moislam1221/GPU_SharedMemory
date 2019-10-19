#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab

GPU = ["TITAN_V", "GTX_1080Ti"]
tpb = [32, 64, 128, 256, 512]
print(len(GPU))
# CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for g in range(len(GPU)):
    pylab.figure()
    for t in range(len(tpb)):
		data = pylab.loadtxt(open("../OVERLAP_RESULTS/" + str(GPU[g]) + "/N1024_TOL1_TPB/" + str(tpb[t]) + ".txt"), delimiter=' ', usecols=(0,1,2,3))
		pylab.semilogy(range(0,tpb[t],2), data[:,1], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
    pylab.xlabel(r'Overlap', fontsize = 20)
    pylab.ylabel(r'Number of Cycles', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.tight_layout()
    pylab.savefig('../FIGURES/overlap_cycles_' + str(GPU[i]) + '.png')

# TIME VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for g in range(len(GPU)):
    pylab.figure()
    for t in range(len(tpb)):
		data = pylab.loadtxt(open("../OVERLAP_RESULTS/" + str(GPU[g]) + "/N1024_TOL1_TPB/" + str(tpb[t]) + ".txt"), delimiter=' ', usecols=(0,1,2,3))
		pylab.semilogy(range(0,tpb[t],2), data[:,2], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
    pylab.xlabel(r'Overlap', fontsize = 20)
    pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.tight_layout()
    pylab.savefig('../FIGURES/overlap_time_' + str(GPU[i]) + '.png')

# TIME/CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for g in range(len(GPU)):
    pylab.figure()
    for t in range(len(tpb)):
		data = pylab.loadtxt(open("../OVERLAP_RESULTS/" + str(GPU[g]) + "/N1024_TOL1_TPB/" + str(tpb[t]) + ".txt"), delimiter=' ', usecols=(0,1,2,3))
		pylab.semilogy(range(0,tpb[t],2), data[:,2] / data[:,1], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
    pylab.xlabel(r'Overlap', fontsize = 20)
    pylab.ylabel(r'Time/Cycle [ms]', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.tight_layout()
    pylab.savefig('../FIGURES/overlap_timepercycle_' + str(GPU[i]) + '.png')

# SPEEDUP VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
for g in range(len(GPU)):
    pylab.figure()
    for t in range(len(tpb)):
		data = pylab.loadtxt(open("../OVERLAP_RESULTS/" + str(GPU[g]) + "/N1024_TOL1_TPB/" + str(tpb[t]) + ".txt"), delimiter=' ', usecols=(0,1,2,3))
		pylab.plot(range(0,tpb[t],2), data[0,1] / data[:,2], '.-', linewidth=2, label = 'tpb = ' + str(tpb[t]))
    pylab.xlabel(r'Overlap', fontsize = 20)
    pylab.ylabel(r'Cycles', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.tight_layout() 
    pylab.savefig('../FIGURES/overlap_speedup_' + str(GPU[i]) + '.png')
