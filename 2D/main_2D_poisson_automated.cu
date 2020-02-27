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
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/solution_error.h"
#include "Helper/setGPU.h"

#define RUN_CPU_FLAG 1
#define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1

// Determine which header files to include based on which directives are active
#ifdef RUN_CPU_FLAG
#include "jacobi-2D-cpu.h"
#endif

#ifdef RUN_GPU_FLAG
#include "jacobi-2D-gpu.h"
#endif

#ifdef RUN_SHARED_FLAG
#include "jacobi-2D-shared.h"
#endif

int main(int argc, char *argv[])
{

    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "TITAN V";
    setGPU(gpuToUse);

	// PARSE INPUTS
    const int nxDim = atoi(argv[1]);
    const int nyDim = atoi(argv[2]);
    const int residual_convergence_metric_flag = atoi(argv[3]);
    const double tolerance_value = atof(argv[4]);
    const int tolerance_reduction_flag = atoi(argv[5]);

    // DEFAULT PARAMETERS FOR ALGORITHM
    const int numTrials = 20;
    const int OVERLAP_X = 0;
    const int OVERLAP_Y = 0;
    const int threadsPerBlock_x = 32; 
    int threadsPerBlock_y;
    int subIterations;

    // INITIALIZE ARRAYS
    int nxGrids = nxDim + 2;
    int nyGrids = nyDim + 2;
    int nDofs = nxGrids * nyGrids;
    double * initX = new double[nDofs];
    double * rhs = new double[nDofs];

    // FILL IN INITIAL CONDITION AND RHS VALUES
    int dof;
    for (int jGrid = 0; jGrid < nyGrids; ++jGrid) {
        for (int iGrid = 0; iGrid < nxGrids; ++iGrid) {
            dof = iGrid + jGrid * nxGrids;
			if (iGrid == 0 || iGrid == nxGrids-1 || jGrid == 0 || jGrid == nyGrids-1) {
				initX[dof] = 0.0f;
			}
			else {
				initX[dof] = 1.0f; 
			}
			rhs[dof] = 1.0f;
        }
    }
 
    // LOAD EXACT SOLUTION IF SOLUTION ERROR IS THE CRITERION FOR CONVERGENCE
    double * solution_exact = new double[nDofs];
    double initSolutionError;
    if (residual_convergence_metric_flag == 0) {
        std::string SOLUTIONEXACT_FILENAME = "solution_exact_N1048576.txt";
        loadSolutionExact(solution_exact, SOLUTIONEXACT_FILENAME, nDofs);
        initSolutionError = solutionError2DPoisson(initX, solution_exact, nDofs);
    }

    // COMPUTE TOLERANCE BASED ON RESIDUAL/ERROR AND INPUTS FROM PYTHON
    double TOL;
    double initResidual = residual2DPoisson(initX, rhs, nxGrids, nyGrids);
    if (tolerance_reduction_flag == 0) {
        TOL = tolerance_value;
    }
    else if (tolerance_reduction_flag == 1 && residual_convergence_metric_flag == 1) {
        TOL = initResidual / tolerance_value;
    }
    else if (tolerance_reduction_flag == 1 && residual_convergence_metric_flag == 0) {
        // TOL = initSolutionError / tolerance_value;
        TOL = (1.0 - 0.01 * tolerance_value) * initSolutionError;
    }

	// CREATE SCRIPT NAMES
    std::string CPU_FILE_NAME = createFileString("CPU", nxDim, nyDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag);
    std::string GPU_FILE_NAME = createFileString("GPU", nxDim, nyDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag);
    std::string SHARED_FILE_NAME = createFileString("SHARED", nxDim, nyDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag);

    int * threadsPerBlock_yarray = new int[64];
    threadsPerBlock_yarray[0] = 4;
    threadsPerBlock_yarray[1] = 8;
    threadsPerBlock_yarray[2] = 16;
    threadsPerBlock_yarray[3] = 32;
    
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of unknowns in x: %d\n", nxDim);
    printf("Number of unknowns in y: %d\n", nyDim);
    printf("Number of Trials: %d\n", numTrials);
    if (residual_convergence_metric_flag == 1) {
    	printf("Residual of initial solution %f\n", initResidual);
    }
    else if (residual_convergence_metric_flag == 0) {
    	printf("Solution Error of initial solution %f\n", initResidual);
    }
    printf("Desired TOL of residual %f\n", TOL);
    printf("======================================================\n");
    
    // CPU - JACOBI
#ifdef RUN_CPU_FLAG
	printf("===============CPU============================\n");
    std::ofstream cpuResults;
    cpuResults.open(CPU_FILE_NAME, std::ios::app);
    int cpuIterations;
    double cpuJacobiTimeTrial;
    double cpuJacobiTimeAverage;
    double cpuTotalTime = 0.0;
    double cpuJacobiResidual;
    double cpuJacobiSolutionError;
	double * solutionJacobiCpu = new double[nDofs];
    if (residual_convergence_metric_flag == 1) {
    	cpuIterations = jacobiCpuIterationCountResidual(initX, rhs, nxGrids, nyGrids, TOL);
    }
    else if (residual_convergence_metric_flag == 0) {
    	cpuIterations = jacobiCpuIterationCountSolutionError(initX, rhs, nxGrids, nyGrids, TOL, solution_exact);
    }
    for (int iter = 0; iter < numTrials; iter++) {
		clock_t cpuJacobiStartTime = clock();
		solutionJacobiCpu = jacobiCpu(initX, rhs, nxGrids, nyGrids, cpuIterations);
		clock_t cpuJacobiEndTime = clock();
		cpuJacobiTimeTrial = (cpuJacobiEndTime - cpuJacobiStartTime) / (double) CLOCKS_PER_SEC;
		cpuJacobiTimeTrial = cpuJacobiTimeTrial * (1e3); // Convert to ms
		cpuTotalTime = cpuTotalTime + cpuJacobiTimeTrial;
        printf("Completed CPU trial %d\n", iter);
    }
    cpuJacobiTimeAverage = cpuTotalTime / numTrials;
    printf("Number of Iterations needed for Jacobi CPU: %d \n", cpuIterations);
    printf("Time needed for the Jacobi CPU: %f ms\n", cpuJacobiTimeAverage);
    if (residual_convergence_metric_flag == 1) {	
		cpuJacobiResidual = residual2DPoisson(solutionJacobiCpu, rhs, nxGrids, nyGrids); 
		printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
        cpuResults << nxDim << " " << nyDim << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiResidual << "\n";
    }
    else if (residual_convergence_metric_flag == 0) {	
		cpuJacobiSolutionError = residual2DPoisson(solutionJacobiCpu, rhs, nxGrids, nyGrids); 
		printf("Solution Error of the Jacobi CPU solution is %f\n", cpuJacobiSolutionError);
        cpuResults << nxDim << " " << nyDim << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiSolutionError << "\n";
    }
    cpuResults.close();
#endif 

    // GPU - JACOBI
#ifdef RUN_GPU_FLAG
	printf("===============GPU============================\n");
	std::ofstream gpuResults;
    gpuResults.open(GPU_FILE_NAME, std::ios::app);
    int gpuIterations;
	float gpuJacobiTimeTrial;
	float gpuJacobiTimeAverage;
	float gputotalTime = 0.0;
    double gpuJacobiResidual;
    double gpuJacobiSolutionError;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double * solutionJacobiGpu = new double[nDofs];
    for (int tpby_idx = 0; tpby_idx < 4; tpby_idx = tpby_idx + 1) {
        threadsPerBlock_y = threadsPerBlock_yarray[tpby_idx];
        if (residual_convergence_metric_flag == 1) {
			gpuIterations = jacobiGpuIterationCountResidual(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y);
		}
        else if (residual_convergence_metric_flag == 0) {
			gpuIterations = jacobiGpuIterationCountSolutionError(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, solution_exact);
		}
		for (int iter = 0; iter < numTrials; iter++) {
			cudaEventRecord(start, 0);
			solutionJacobiGpu = jacobiGpu(initX, rhs, nxGrids, nyGrids, gpuIterations, threadsPerBlock_x, threadsPerBlock_y);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&gpuJacobiTimeTrial, start, stop);
			gputotalTime = gputotalTime + gpuJacobiTimeTrial;
			printf("Threads Per Block in x = %d and Threads Per Block in y = %d: Completed GPU trial %d\n", threadsPerBlock_x, threadsPerBlock_y, iter);
		}
    	gpuJacobiTimeAverage = gputotalTime / numTrials;
        printf("Number of Iterations needed for Jacobi GPU: %d \n", gpuIterations);
        printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTimeAverage);
		if (residual_convergence_metric_flag == 1) {
			gpuJacobiResidual = residual2DPoisson(solutionJacobiGpu, rhs, nxGrids, nyGrids); 
            printf("Residual of the Jacobi GPU solution is %f\n", gpuJacobiResidual);
            gpuResults << nxDim << " " << nyDim << " " << threadsPerBlock_x << " " << threadsPerBlock_y << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiResidual << "\n";
        }
        else if (residual_convergence_metric_flag == 0) {
            gpuJacobiSolutionError = solutionError2DPoisson(solutionJacobiGpu, solution_exact, nDofs);
            printf("Solution Error of the Jacobi GPU solution is %f\n", gpuJacobiSolutionError);
            gpuResults << nxDim << " " << nyDim << " " << threadsPerBlock_x << " " << threadsPerBlock_y << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiSolutionError << "\n";
        }
        gputotalTime = 0.0;
    }
    gpuResults.close();
#endif

    // SHARED - JACOBI
#ifdef RUN_SHARED_FLAG
    printf("===============SHARED============================\n");
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    std::ofstream sharedResults;
    sharedResults.open(SHARED_FILE_NAME, std::ios::app);
    int sharedCycles;
    float sharedJacobiTimeTrial;
    float sharedJacobiTimeAverage;
    float sharedtotalTime = 0.0;
    double sharedJacobiResidual;
    double sharedJacobiSolutionError;
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    double * solutionJacobiShared = new double[nDofs];
 	for (int tpby_idx = 0; tpby_idx < 4; tpby_idx = tpby_idx + 1) {
        threadsPerBlock_y = threadsPerBlock_yarray[tpby_idx];
    	subIterations = std::min(threadsPerBlock_x, threadsPerBlock_y) / 2;
        if (residual_convergence_metric_flag == 1) {
            sharedCycles = jacobiSharedIterationCountResidual(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations);
        }
        else if (residual_convergence_metric_flag == 0) {
            sharedCycles = jacobiSharedIterationCountSolutionError(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations, solution_exact);
		}
        for (int iter = 0; iter < numTrials; iter++) {
            cudaEventRecord(start_sh, 0);
            solutionJacobiShared = jacobiShared(initX, rhs, nxGrids, nyGrids, sharedCycles, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations);
            cudaEventRecord(stop_sh, 0);
            cudaEventSynchronize(stop_sh);
            cudaEventElapsedTime(&sharedJacobiTimeTrial, start_sh, stop_sh);
            sharedtotalTime = sharedtotalTime + sharedJacobiTimeTrial;
        	printf("Threads Per Block in x = %d and Threads Per Block in y = %d: Completed GPU trial %d\n", threadsPerBlock_x, threadsPerBlock_y, iter);
        }
        sharedJacobiTimeAverage = sharedtotalTime / numTrials;
        printf("Number of Cycles needed for Jacobi Shared: %d (%d) \n", sharedCycles, subIterations);
        printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTimeAverage);
        if (residual_convergence_metric_flag == 1) {
            sharedJacobiResidual = residual2DPoisson(solutionJacobiShared, rhs, nxGrids, nyGrids);
            printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
            sharedResults << nxDim << " " << nyDim << " " << threadsPerBlock_x << " " << threadsPerBlock_y << " " << numTrials << " " << sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiResidual << "\n";
        }
        else if (residual_convergence_metric_flag == 0) {
            sharedJacobiSolutionError = solutionError2DPoisson(solutionJacobiShared, solution_exact, nDofs);
            printf("Solution Error of the Jacobi Shared solution is %f\n", sharedJacobiSolutionError);
            sharedResults << nxDim << " " << nyDim << " " << threadsPerBlock_x << " " << threadsPerBlock_y << " " << numTrials << " " << sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiSolutionError << "\n";
        }
        sharedtotalTime = 0.0;
    }
    sharedResults.close();
#endif

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solution_exact;
    delete[] threadsPerBlock_yarray;
#ifdef RUN_CPU_FLAG
    delete[] solutionJacobiCpu;
#endif 
#ifdef RUN_GPU_FLAG 
    delete[] solutionJacobiGpu;
#endif
#ifdef RUN_SHARED_FLAG 
    delete[] solutionJacobiShared;
#endif
    
    return 0;
}
