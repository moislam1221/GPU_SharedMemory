#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import numpy as np
import pylab

# LOAD DATA FROM FILES
tpb32Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB32.txt"), delimiter=' ', usecols=(0,1,2,3))
tpb64Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB64.txt"), delimiter=' ', usecols=(0,1,2,3))
tpb128Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB128.txt"), delimiter=' ', usecols=(0,1,2,3))
tpb256Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB256.txt"), delimiter=' ', usecols=(0,1,2,3))
tpb512Data = pylab.loadtxt(open("../OVERLAP_RESULTS/N1024_TOL1_TPB512.txt"), delimiter=' ', usecols=(0,1,2,3))

tpb = [32, 64, 128, 256, 512]

# TIME VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
pylab.plot(range(0,tpb[0],2), tpb32Data[:,2], '.-', linewidth=2, label = 'tpb = 32')
pylab.plot(range(0,tpb[1],2), tpb64Data[:,2], '.-', linewidth=2, label = 'tpb = 64')
pylab.plot(range(0,tpb[2],2), tpb128Data[:,2], '.-', linewidth=2, label = 'tpb = 128')
pylab.plot(range(0,tpb[3],2), tpb256Data[:,2], '.-', linewidth=2, label = 'tpb = 256')
pylab.plot(range(0,tpb[4],2), tpb512Data[:,2], '.-', linewidth=2, label = 'tpb = 512')
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Time To Achieve TOL [ms]', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/overlap_time.png')

# CYCLES VS OVERLAP FOR VARIOUS TPB (= 32, 64, 128, 256, 1024)
pylab.figure()
pylab.plot(range(0,tpb[0],2), tpb32Data[:,1], '.-', linewidth=2, label = 'tpb = 32')
pylab.plot(range(0,tpb[1],2), tpb64Data[:,1], '.-', linewidth=2, label = 'tpb = 64')
pylab.plot(range(0,tpb[2],2), tpb128Data[:,1], '.-', linewidth=2, label = 'tpb = 128')
pylab.plot(range(0,tpb[3],2), tpb256Data[:,1], '.-', linewidth=2, label = 'tpb = 256')
pylab.plot(range(0,tpb[4],2), tpb512Data[:,1], '.-', linewidth=2, label = 'tpb = 512')
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Number of Cycles', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/overlap_cycles.png')

# SPEEDUP COMPARED TO NO OVERLAP CASE
pylab.figure()
pylab.plot(range(0,tpb[0],2), tpb32Data[0,2]/tpb32Data[:,2], '.-', linewidth=2, label = 'tpb = 32')
pylab.plot(range(0,tpb[1],2), tpb64Data[0,2]/tpb64Data[:,2], '.-', linewidth=2, label = 'tpb = 64')
pylab.plot(range(0,tpb[2],2), tpb128Data[0,2]/tpb128Data[:,2], '.-', linewidth=2, label = 'tpb = 128')
pylab.plot(range(0,tpb[3],2), tpb256Data[0,2]/tpb256Data[:,2], '.-', linewidth=2, label = 'tpb = 256')
pylab.plot(range(0,tpb[4],2), tpb512Data[0,2]/tpb512Data[:,2], '.-', linewidth=2, label = 'tpb = 512')
pylab.xlabel(r'Overlap', fontsize = 20)
pylab.ylabel(r'Speedup (Relative to No Overlap)', fontsize = 18)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.tight_layout()
pylab.savefig('../FIGURES/overlap_speedup.png')
