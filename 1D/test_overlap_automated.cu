#include<utility>
#include<stdio.h>
#include<assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <utility>
#include <time.h>

// HEADER FILES
#include "Helper/createFileStrings.h"
#include "Helper/fillThreadsPerBlock.h"
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/solution_error.h"
#include "Helper/setGPU.h"

// Header file for shared jacobi code
#include "jacobi-1D-shared.h"

int main(int argc, char *argv[])
{
    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "TITAN V";
    setGPU(gpuToUse);

    // PARSE INPUTS
    const int nDim = atoi(argv[1]);
    const int residual_convergence_metric_flag = atoi(argv[2]);
    const int tolerance_value = atoi(argv[3]);
    const int tolerance_reduction_flag = atoi(argv[4]);

    // DEFAULT PARAMETERS
    const int numTrials = 20;
    int threadsPerBlock, subIterations;
 
    // INITIALIZE ARRAYS
    int nGrids = nDim + 2;
    double * initX = new double[nGrids];
    double * rhs = new double[nGrids];
    
    // 1D POISSON MATRIX
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        if (iGrid == 0 || iGrid == nGrids-1) {
            initX[iGrid] = 0.0f;
        }
        else {
            initX[iGrid] = 1.0f; 
        }
        rhs[iGrid] = 1.0f;
    }

    // LOAD EXACT SOLUTION IF SOLUTION ERROR IS THE CRITERION FOR CONVERGENCE
    double * solution_exact = new double[nGrids];
    double initSolutionError;
    if (residual_convergence_metric_flag == 0) {
        std::string SOLUTIONEXACT_FILENAME = "solution_exact_N32.txt";
        loadSolutionExact(solution_exact, SOLUTIONEXACT_FILENAME, nGrids);
        initSolutionError = solutionError1DPoisson(initX, solution_exact, nGrids);
    }

    // COMPUTE TOLERANCE BASED ON RESIDUAL/ERROR AND INPUTS FROM PYTHON
    double TOL;
    double initResidual = residual1DPoisson(initX, rhs, nGrids);
    if (tolerance_reduction_flag == 0) {
        TOL = tolerance_value;
    }
    else if (tolerance_reduction_flag == 1 && residual_convergence_metric_flag == 1) {
        TOL = initResidual / tolerance_value;
    }
    else if (tolerance_reduction_flag == 1 && residual_convergence_metric_flag == 0) {
        TOL = initSolutionError / tolerance_value;
    }

    // THREADS PER BLOCK VALUES
    int* threadsPerBlock_array = new int[6];
    fillThreadsPerBlockArray(threadsPerBlock_array);
    
    // DEFINE CUDA EVENTS
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    
    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of unknowns: %d\n", nDim);
    printf("Number of Trials: %d\n", numTrials);
    if (residual_convergence_metric_flag == 1) {
        printf("Residual of initial solution: %f\n", initResidual);
    }
    else if (residual_convergence_metric_flag == 0) {
        printf("Solution Error of initial solution: %f\n", initSolutionError);
    }
    printf("Desired TOL of residual/solution error: %f\n", TOL);
    printf("======================================================\n");

    // NECESSARY CONTAINERS
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    int OVERLAP;
    int numIterations;
    float sharedJacobiTime;
    float totalTime = 0.0;
    std::string SHARED_FILE_NAME;
    std::ofstream timings_sh;
    for (int tpb_idx = 0; tpb_idx < 6; tpb_idx = tpb_idx + 1) {
        threadsPerBlock = threadsPerBlock_array[tpb_idx];
		numIterations = threadsPerBlock / 2;
		subIterations = threadsPerBlock / 2;
    	SHARED_FILE_NAME = createFileStringOverlap(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag);
        std::cout << SHARED_FILE_NAME << std::endl;
    	timings_sh.open(SHARED_FILE_NAME.c_str(), std::ios_base::app);
		int * sharedCycles = new int[numIterations];
		double * sharedJacobiTimeArray = new double[numIterations];
		double * sharedJacobiResidual = new double[numIterations];
		double * sharedJacobiSolutionError = new double[numIterations];
		double * solutionJacobiShared = new double[nGrids];
		for (int i = 0; i < numIterations; i++) {
			// OBTAIN NUMBER OF CYCLES TO CONVERGE FOR GIVEN OVERLAP
			OVERLAP = 2*i;
			if (residual_convergence_metric_flag == 1) {
				sharedCycles[i] = jacobiSharedIterationCountResidual(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
            }
			else if (residual_convergence_metric_flag == 0) {
				sharedCycles[i] = jacobiSharedIterationCountSolutionError(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations, solution_exact);
            }
			for (int iter = 0; iter < numTrials; iter++) {
				//if (threadsPerBlock == innerSubdomainLength) {
					// GET FINAL SOLUTION
					cudaEventRecord(start_sh, 0);
					solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles[i], threadsPerBlock, OVERLAP, subIterations);
					// OBTAIN FINAL TIMES REQUIRED
					cudaEventRecord(stop_sh, 0);
					cudaEventSynchronize(stop_sh);
					cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
					totalTime = totalTime + sharedJacobiTime;
					printf("THREADS PER BLOCK: %d, OVERLAP = %d/%d, TRIAL %d/%d\n", threadsPerBlock, OVERLAP, threadsPerBlock-2, iter, numTrials);
			}
			sharedJacobiTimeArray[i] = totalTime / numTrials;
			printf("Number of Cycles: %d (subiterations = %d) \n", sharedCycles[i], threadsPerBlock/2);
			printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTimeArray[i]);
			if (residual_convergence_metric_flag == 1) {
				sharedJacobiResidual[i] = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
				printf("Residual is %f\n", sharedJacobiResidual[i]);
				timings_sh << OVERLAP << " " <<  sharedCycles[i] << " " << sharedJacobiTimeArray[i] << " " << sharedJacobiResidual[i] << " " << numTrials << " " << "\n";
			}
			else if (residual_convergence_metric_flag == 0) {
				sharedJacobiSolutionError[i] = solutionError1DPoisson(solutionJacobiShared, solution_exact, nGrids);
				printf("Solution Error is %f\n", sharedJacobiSolutionError[i]);
				timings_sh << OVERLAP << " " <<  sharedCycles[i] << " " << sharedJacobiTimeArray[i] << " " << sharedJacobiSolutionError[i] << " " << numTrials << " " << "\n";
			}
			printf("================================================\n");
			totalTime = 0.0;
		}    
    	timings_sh.close();
   		delete[] sharedCycles; 
   		delete[] sharedJacobiTimeArray; 
   		delete[] sharedJacobiResidual; 
   		delete[] sharedJacobiSolutionError; 
   		delete[] solutionJacobiShared; 
	} 

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solution_exact;
    delete[] threadsPerBlock_array;

    return 0;
}
