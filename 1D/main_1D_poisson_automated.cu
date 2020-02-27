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
#include <iomanip>

// HEADER FILES
#include "Helper/createFileStrings.h"
#include "Helper/fillThreadsPerBlock.h"
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/solution_error.h"
#include "Helper/setGPU.h"

#define RUN_CPU_FLAG 1
#define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1

// Determine which header files to include based on which directives are active
#ifdef RUN_CPU_FLAG
#include "jacobi-1D-cpu.h"
#endif

#ifdef RUN_GPU_FLAG
#include "jacobi-1D-gpu.h"
#endif

#ifdef RUN_SHARED_FLAG
#include "jacobi-1D-shared.h"
#endif

int main(int argc, char *argv[])
{
    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    // Supercloud has V100s that we'll want to use soon
    std::string gpuToUse = "TITAN V"; 
    setGPU(gpuToUse);
    
    // PARSE INPUTS
    const int nDim = atoi(argv[1]); 
    const int residual_convergence_metric_flag = atoi(argv[2]);
    const double tolerance_value = atof(argv[3]);
    const int tolerance_reduction_flag = atoi(argv[4]);
	const int relaxation_flag = 1;

    // DEFAULT PARAMETERS FOR ALGORITHM
    const int OVERLAP = 0;
    const int numTrials = 20;
    int threadsPerBlock, subIterations;

    // INITIALIZE ARRAYS
    int nGrids = nDim + 2;
    double * initX = new double[nGrids];
    double * rhs = new double[nGrids];
    
    // FILL IN INITIAL CONDITION AND RHS VALUES
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
        std::string SOLUTIONEXACT_FILENAME = "solution_exact_N1048576.txt";
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
        // TOL = initSolutionError / tolerance_value;
        TOL = (1.0 - 0.01 * tolerance_value) * initSolutionError;
    }
/*
    // CREATE COMPONENTS OF SCRIPT NAMES BASED ON INPUTS
    std::string CPU_BASE_NAME = "RESULTS/CPU.";
    std::string GPU_BASE_NAME = "RESULTS/GPU.";
    std::string SHARED_BASE_NAME = "RESULTS/SHARED.";
    std::string N_STRING = "N" + std::to_string(nDim) + ".";
    std::string TOL_TYPE_STRING;
    if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "TOLREDUCE";
    }
    else if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "TOL";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "ERRORREDUCE";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "ERROR";

    }
    std::string TOL_VAL_STRING = std::to_string(tolerance_value);
    std::string TXT_STRING = ".txt";
    
    // CREATE CPU/GPU/SHARED STRING NAMES
    std::string CPU_FILE_NAME = CPU_BASE_NAME + N_STRING + TOL_TYPE_STRING + TOL_VAL_STRING + TXT_STRING;
    std::string GPU_FILE_NAME = GPU_BASE_NAME + N_STRING + TOL_TYPE_STRING + TOL_VAL_STRING + TXT_STRING;
    std::string SHARED_FILE_NAME = SHARED_BASE_NAME + N_STRING + TOL_TYPE_STRING + TOL_VAL_STRING + TXT_STRING;
*/	
    
    std::string CPU_FILE_NAME = createFileString("CPU", nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag, relaxation_flag);
    std::string GPU_FILE_NAME = createFileString("GPU", nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag, relaxation_flag);
    std::string SHARED_FILE_NAME = createFileString("SHARED", nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag, relaxation_flag);

    std::cout << CPU_FILE_NAME << std::endl;
    std::cout << GPU_FILE_NAME << std::endl;
    std::cout << SHARED_FILE_NAME << std::endl;
    
    int * threadsPerBlock_array = new int[6];
    fillThreadsPerBlockArray(threadsPerBlock_array);

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
 
    // CPU - JACOBI
#ifdef RUN_CPU_FLAG
	printf("===============CPU============================\n");
	std::ofstream cpuResults;
	cpuResults.open(CPU_FILE_NAME, std::ios::app);
    int cpuIterations;
    float cpuJacobiTimeTrial;
    float cpuJacobiTimeAverage;
    float cpuTotalTime = 0.0;
    double cpuJacobiResidual;
    double cpuJacobiSolutionError;
    double * solutionJacobiCpu = new double[nGrids];
    if (residual_convergence_metric_flag == 1) {
	    cpuIterations = jacobiCpuIterationCountResidual(initX, rhs, nGrids, TOL);
    }
    else if (residual_convergence_metric_flag == 0) {
	    cpuIterations = jacobiCpuIterationCountSolutionError(initX, rhs, nGrids, TOL, solution_exact);
    }
    for (int iter = 0; iter < numTrials; iter++) {
		clock_t cpuJacobiStartTime = clock();
		solutionJacobiCpu = jacobiCpu(initX, rhs, nGrids, cpuIterations);
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
	    cpuJacobiResidual = residual1DPoisson(solutionJacobiCpu, rhs, nGrids);
	    printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
	    cpuResults << nDim << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiResidual << "\n";
    }
	else if (residual_convergence_metric_flag == 0) {
        cpuJacobiSolutionError = solutionError1DPoisson(solutionJacobiCpu, solution_exact, nGrids);
	    printf("Solution Error of the Jacobi CPU solution is %f\n", cpuJacobiSolutionError);
	    cpuResults << nDim << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiSolutionError << "\n";
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
    double * solutionJacobiGpu = new double[nGrids];
    for (int tpb_idx = 0; tpb_idx < 6; tpb_idx = tpb_idx + 1) {
        threadsPerBlock = threadsPerBlock_array[tpb_idx];
        if (residual_convergence_metric_flag == 1) {
            gpuIterations = jacobiGpuIterationCountResidual(initX, rhs, nGrids, TOL, threadsPerBlock);
        }
        else if (residual_convergence_metric_flag == 0) {
            gpuIterations = jacobiGpuIterationCountSolutionError(initX, rhs, nGrids, TOL, threadsPerBlock, solution_exact);
        }
        for (int iter = 0; iter < numTrials; iter++) {
            cudaEventRecord(start, 0);
            solutionJacobiGpu = jacobiGpu(initX, rhs, nGrids, gpuIterations, threadsPerBlock);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpuJacobiTimeTrial, start, stop);
            // printf("Time = %f\n", gpuJacobiTimeTrial);
            gputotalTime = gputotalTime + gpuJacobiTimeTrial;
            printf("Threads Per Block = %d: Completed GPU trial %d\n", threadsPerBlock, iter);
        }
        gpuJacobiTimeAverage = gputotalTime / numTrials;
	    printf("Number of Iterations needed for Jacobi GPU: %d \n", gpuIterations);
    	printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTimeAverage);
        if (residual_convergence_metric_flag == 1) {
            gpuJacobiResidual = residual1DPoisson(solutionJacobiGpu, rhs, nGrids);
	        printf("Residual of the Jacobi GPU solution is %f\n", gpuJacobiResidual);
            gpuResults << nDim << " " << threadsPerBlock << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiResidual << "\n";
        }
        else if (residual_convergence_metric_flag == 0) {
            gpuJacobiSolutionError = solutionError1DPoisson(solutionJacobiGpu, solution_exact, nGrids);
	        printf("Solution Error of the Jacobi GPU solution is %f\n", gpuJacobiSolutionError);
            gpuResults << nDim << " " << threadsPerBlock << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiSolutionError << "\n";
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
    double * solutionJacobiShared = new double[nGrids];
    for (int tpb_idx = 0; tpb_idx < 6; tpb_idx = tpb_idx + 1) {
        threadsPerBlock = threadsPerBlock_array[tpb_idx];
        subIterations = threadsPerBlock / 2;
        if (residual_convergence_metric_flag == 1) {
            sharedCycles = jacobiSharedIterationCountResidual(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
        }
        else if (residual_convergence_metric_flag == 0) {
            sharedCycles = jacobiSharedIterationCountSolutionError(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations, solution_exact);
        }
        for (int iter = 0; iter < numTrials; iter++) {
            cudaEventRecord(start_sh, 0);
            solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles, threadsPerBlock, OVERLAP, subIterations);
            cudaEventRecord(stop_sh, 0);
            cudaEventSynchronize(stop_sh);
            cudaEventElapsedTime(&sharedJacobiTimeTrial, start_sh, stop_sh);
            sharedtotalTime = sharedtotalTime + sharedJacobiTimeTrial;
            printf("Completed GPU Shared trial %d for threads per block of %d\n", iter, threadsPerBlock);
        }
        sharedJacobiTimeAverage = sharedtotalTime / numTrials;
        printf("Number of Cycles needed for Jacobi Shared: %d (%d) \n", sharedCycles, threadsPerBlock/2);
        printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTimeAverage);
        if (residual_convergence_metric_flag == 1) {
            sharedJacobiResidual = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
            printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
            sharedResults << nDim << " " << threadsPerBlock << " " << numTrials << " " << sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiResidual << "\n";
        }
        else if (residual_convergence_metric_flag == 0) {
            sharedJacobiSolutionError = solutionError1DPoisson(solutionJacobiShared, solution_exact, nGrids);
            printf("Solution Error of the Jacobi Shared solution is %f\n", sharedJacobiSolutionError);
            sharedResults << nDim << " " << threadsPerBlock << " " << numTrials << " " << sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiSolutionError << "\n";
        }
        sharedtotalTime = 0.0;
    }
	sharedResults.close();
#endif

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solution_exact;
    delete[] threadsPerBlock_array;
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
