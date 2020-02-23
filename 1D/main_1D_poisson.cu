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
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/solution_error.h"
#include "Helper/setGPU.h"

// #define RUN_CPU_FLAG 1
#define RUN_GPU_FLAG 1
// #define RUN_SHARED_FLAG 1

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
    // std::string gpuToUse = "Quadro K420"; 
    // setGPU(gpuToUse);
    
    // INPUTS AND OUTPUT FILE NAMES
    const int nDim = 1024; // 4096; // 65536; //524288; //65536; //atoi(argv[1]); 
    const int threadsPerBlock = 1024; //32; //512; // 32; 
    const double TOL = 1.0; //atoi(argv[4]);
    // const double residualReductionFactor = 10000.0; //atoi(argv[4]);
    // const double errorReductionFactor = 0.95; // 10000000.0; //atoi(argv[4]);
    const int OVERLAP = 0;
    const int subIterations = threadsPerBlock / 2;
    const int numTrials = 20;
    std::string CPU_FILE_NAME = "RESULTS/CPU_N1024_TOL1_DOUBLES.txt";
    std::string GPU_FILE_NAME = "RESULTS/GPU_N1024_TOL1_DOUBLES.txt";
    std::string SHARED_FILE_NAME = "RESULTS/SHARED_N1024_TOL1_DOUBLES.txt"; 
    // SHARED_N1024_ERRORREDUCE100.txt";
    /////////////////////////////////////////////////////////////////////////

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

    // LOAD EXACT SOLUTION
    // double * solution_exact = new double[nGrids];
    // std::string SOLUTIONEXACT_FILENAME = "solution_exact_N65536_long.txt";
    // loadSolutionExact(solution_exact, SOLUTIONEXACT_FILENAME, nGrids);

    /*for (int i = 1; i < nGrids; ++i) {
        initX[i] = solution_exact[i];
    }*/

    // COMPUTE INITIAL RESIDUAL AND SET TOLERANCE
	double initResidual = residual1DPoisson(initX, rhs, nGrids);
	// double initSolutionError = solutionError1DPoisson(initX, solution_exact, nGrids);
    // const double TOL = initResidual / residualReductionFactor; //atoi(argv[4]);
    // const double TOL = initSolutionError * errorReductionFactor; // initSolutionError / errorReductionFactor; //atoi(argv[4]);
    
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    // printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Residual of initial solution: %f\n", initResidual);
    // printf("Solution Error of initial solution: %f\n", initSolutionError);
    printf("Desired TOL of residual/solution error: %f\n", TOL);
    // printf("Residual reduction factor: %f\n", errorReductionFactor);
    printf("Number of Trials: %d\n", numTrials);
    printf("======================================================\n");
    
    // CPU - JACOBI
#ifdef RUN_CPU_FLAG
	int cpuIterations = jacobiCpuIterationCountResidual(initX, rhs, nGrids, TOL);
	// int cpuIterations = jacobiCpuIterationCount(initX, solution_exact, rhs, nGrids, TOL);
    float cpuJacobiTimeTrial;
    float cpuJacobiTimeAverage;
    float cpuTotalTime = 0.0;
    double cpuJacobiResidual;
    double cpuJacobiSolutionError;
    double * solutionJacobiCpu;
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
	cpuJacobiResidual = residual1DPoisson(solutionJacobiCpu, rhs, nGrids);
	// cpuJacobiSolutionError = solutionError1DPoisson(solutionJacobiCpu, solution_exact, nGrids);
#endif

    // GPU - JACOBI
#ifdef RUN_GPU_FLAG
	int gpuIterations = jacobiGpuIterationCountResidual(initX, rhs, nGrids, TOL, threadsPerBlock);
	// int gpuIterations = jacobiGpuIterationCount(initX, solution_exact, rhs, nGrids, TOL, threadsPerBlock);
	float gpuJacobiTimeTrial;
	float gpuJacobiTimeAverage;
    float gputotalTime = 0.0;
	double gpuJacobiResidual;
	double gpuJacobiSolutionError;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    double * solutionJacobiGpu;
    for (int iter = 0; iter < numTrials; iter++) {
		cudaEventRecord(start, 0);
		solutionJacobiGpu = jacobiGpu(initX, rhs, nGrids, gpuIterations, threadsPerBlock);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpuJacobiTimeTrial, start, stop);
        gputotalTime = gputotalTime + gpuJacobiTimeTrial;
        printf("Completed GPU trial %d\n", iter);
	}
    gpuJacobiTimeAverage = gputotalTime / numTrials;
    gpuJacobiResidual = residual1DPoisson(solutionJacobiGpu, rhs, nGrids);
    // gpuJacobiSolutionError = solutionError1DPoisson(solutionJacobiGpu, solution_exact, nGrids);
#endif
 
    // SHARED - JACOBI
#ifdef RUN_SHARED_FLAG
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	// int sharedCycles = jacobiSharedIterationCountSolutionError(initX, solution_exact, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
	int sharedCycles = jacobiSharedIterationCountResidual(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
    float sharedJacobiTimeTrial;
    float sharedJacobiTimeAverage;
    float sharedtotalTime = 0;
    double sharedJacobiResidual;
    double sharedJacobiSolutionError;
	cudaEvent_t start_sh, stop_sh;
	cudaEventCreate(&start_sh);
	cudaEventCreate(&stop_sh);
    double * solutionJacobiShared;
    for (int iter = 0; iter < numTrials; iter++) {
        cudaEventRecord(start_sh, 0);
	    solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles, threadsPerBlock, OVERLAP, subIterations);
	    cudaEventRecord(stop_sh, 0);
	    cudaEventSynchronize(stop_sh);
	    cudaEventElapsedTime(&sharedJacobiTimeTrial, start_sh, stop_sh);
        sharedtotalTime = sharedtotalTime + sharedJacobiTimeTrial;
        printf("Completed GPU Shared trial %d\n", iter);
	}
    sharedJacobiTimeAverage = sharedtotalTime / numTrials;
    sharedJacobiResidual = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
    // sharedJacobiSolutionError = solutionError1DPoisson(solutionJacobiShared, solution_exact, nGrids);
#endif

    double dx = 1.0 / (nGrids - 1);
    std::cout << std::fixed << std::setprecision(10) << dx*dx << std::endl;
    double * residualArray = new double[nGrids];
    double * residualArrayExact = new double[nGrids];
    residualArray[0] = 0.0;
    residualArray[nGrids-1] = 0.0;
    /* for (int i = 1; i < nGrids-1; i++) {
        // residualArray[i] = abs(rhs[i] + (solutionJacobiCpu[i-1] - 2.0*solutionJacobiCpu[i] +  solutionJacobiCpu[i+1]) / (dx*dx));
        // residualArray[i] = abs(rhs[i] + (solutionJacobiGpu[i-1] - 2.0*solutionJacobiGpu[i] +  solutionJacobiGpu[i+1]) / (dx*dx));
        residualArray[i] = abs(rhs[i] + (solutionJacobiShared[i-1] - 2.0*solutionJacobiShared[i] +  solutionJacobiShared[i+1]) / (dx*dx));
        residualArrayExact[i] = abs(rhs[i] + (solution_exact[i-1] - 2.0*solution_exact[i] + solution_exact[i+1]) / (dx*dx));
    }*/
 
    // PRINT SOLUTION - NEEDS ADJUSTING BASED ON WHICH FLAGS ARE ON
    for (int i = 0; i < nGrids; i++) {
        // printf("Grid %d = %f\n", i, solutionJacobiShared[i]);
        // std::cout << std::fixed << std::setprecision(10) << solution_exact[i] << "\t" << solutionJacobiCpu[i] << "\t" << abs(solution_exact[i] - solutionJacobiCpu[i]) << "\t" << residualArray[i] << "\t" << residualArrayExact[i] << std::endl;
        // std::cout << std::fixed << std::setprecision(10) << solution_exact[i] << "\t" << solutionJacobiGpu[i] << "\t" << abs(solution_exact[i] - solutionJacobiGpu[i]) << "\t" << residualArray[i] << "\t" << residualArrayExact[i] << std::endl;
        // std::cout << std::fixed << std::setprecision(10) << solution_exact[i] << "\t" << solutionJacobiShared[i] << "\t" << abs(solution_exact[i] - solutionJacobiShared[i]) << "\t" << residualArray[i] << "\t" << residualArrayExact[i] << std::endl;
    }
    
    
    // CPU RESULTS
#ifdef RUN_CPU_FLAG 
	printf("===============CPU============================\n");
	printf("Number of Iterations needed for Jacobi CPU: %d \n", cpuIterations);
	printf("Time needed for the Jacobi CPU: %f ms\n", cpuJacobiTimeAverage);
	printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
	// printf("Solution Error of the Jacobi CPU solution is %f\n", cpuJacobiSolutionError);
	std::ofstream cpuResults;
	cpuResults.open(CPU_FILE_NAME, std::ios::app);
	// cpuResults << nDim << " " << residualReductionFactor << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiResidual << "\n";
	// cpuResults << nDim << " " << errorReductionFactor << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiSolutionError << "\n";
	cpuResults << nDim << " " << " "  << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiResidual << "\n";
	cpuResults.close();
#endif
 
    // GPU RESULTS
#ifdef RUN_GPU_FLAG
	printf("===============GPU============================\n");
	printf("Number of Iterations needed for Jacobi GPU: %d \n", gpuIterations);
	printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTimeAverage);
	printf("Residual of the Jacobi GPU solution is %f\n", gpuJacobiResidual);
	// printf("Solution Error of the Jacobi GPU solution is %f\n", gpuJacobiSolutionError);
	std::ofstream gpuResults;
	gpuResults.open(GPU_FILE_NAME, std::ios::app);
	// gpuResults << nDim << " " << threadsPerBlock << " " << residualReductionFactor << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiResidual << "\n";
	// gpuResults << nDim << " " << threadsPerBlock << " " << errorReductionFactor << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiSolutionError << "\n";
	gpuResults << nDim << " " << threadsPerBlock << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiResidual << "\n";
	gpuResults.close();
#endif

    // SHARED RESULTS
#ifdef RUN_SHARED_FLAG 
	printf("===============SHARED============================\n");
	printf("Number of Cycles needed for Jacobi Shared: %d (%d) \n", sharedCycles, threadsPerBlock/2);
	printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTimeAverage);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
	// printf("Solution Error of the Jacobi Shared solution is %f\n", sharedJacobiSolutionError);
	std::ofstream sharedResults;
	sharedResults.open(SHARED_FILE_NAME, std::ios::app);
	// sharedResults << nDim << " " << threadsPerBlock << " " << residualReductionFactor << " " << numTrials << " " <<  sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiResidual << "\n";
	// sharedResults << nDim << " " << threadsPerBlock << " " << errorReductionFactor << " " << numTrials << " " << sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiSolutionError << "\n";
	sharedResults << nDim << " " << threadsPerBlock << " " << numTrials << " " << sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiResidual << "\n";
	sharedResults.close();
#endif

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    // delete[] solution_exact;
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
