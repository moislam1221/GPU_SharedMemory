#!/usr/bin/env python

# IMPORT PACKAGES
import os
import sys
sys.path.insert(0, 'Helper')
from createFileStrings import figuresSubfolderName

# PARSE INPUTS
nDim = sys.argv[1]
residual_convergence_metric_flag = sys.argv[2]
tolerance_value = sys.argv[3]
tolerance_reduction_flag = sys.argv[4]

# CREATE THE FIGURES SUBFOLDER FOLDERFIGURES_SUBFOLDER = figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)
SUBFOLDER = figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag)

# CREATE NECESSARY FOLDERS
pathName = "FIGURES/" + SUBFOLDER
if not os.path.exists(pathName):
    print(pathName)
    os.makedirs(pathName)
pathName = "FIGURES/" + SUBFOLDER + "/INITIAL"
if not os.path.exists(pathName):
    os.makedirs(pathName)
pathName = "FIGURES/" + SUBFOLDER + "/OVERLAP_RESULTS"
if not os.path.exists(pathName):
    os.makedirs(pathName)
# CREATE NECESSARY FOLDERS
pathName = "FIGURES/" + SUBFOLDER + "/SUBITERATION_RESULTS"
if not os.path.exists(pathName):
    os.makedirs(pathName)
# CREATE NECESSARY FOLDERS
pathName = "FIGURES/" + SUBFOLDER + "/OPTIMIZED"
if not os.path.exists(pathName):
    os.makedirs(pathName)

