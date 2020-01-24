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
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/setGPU.h"
#include "jacobi-1D-cpu.h"
#include "jacobi-1D-gpu.h"
#include "jacobi-1D-shared.h"

// #define RUN_CPU_FLAG 1
// #define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1

int main(int argc, char *argv[])
{
    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "TITAN V"; 
    setGPU(gpuToUse);

    // INPUTS AND OUTPUT FILE NAMES
    const int nDim = 1048576; // 65536; //524288; //65536; //atoi(argv[1]); 
    const int threadsPerBlock = 1024; //32; //512; // 32; 
    // const float TOL = 1.0; //atoi(argv[4]);
    const float residualReductionFactor = 10000.0; //atoi(argv[4]);
    const int OVERLAP = 0;
    const int subIterations = threadsPerBlock / 2;
    const int numTrials = 20;
    std::string CPU_FILE_NAME = "RESULTS/CPU_N1048576_TOLREDUCE10000.txt";
    std::string GPU_FILE_NAME = "RESULTS/GPU_N1048576_TOLREDUCE10000.txt";
    std::string SHARED_FILE_NAME = "RESULTS/SHARED_N1048576_TOLREDUCE10000.txt";
    /////////////////////////////////////////////////////////////////////////

    // INITIALIZE ARRAYS
    int nGrids = nDim + 2;
    float * initX = new float[nGrids];
    float * rhs = new float[nGrids];
    
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

    // COMPUTE INITIAL RESIDUAL AND SET TOLERANCE
	float initResidual = residual1DPoisson(initX, rhs, nGrids);
    const float TOL = initResidual / residualReductionFactor; //atoi(argv[4]);
    
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Residual of initial solution: %f\n", initResidual);
    printf("Desired TOL of residual: %f\n", TOL);
    printf("Residual reduction factor: %f\n", residualReductionFactor);
    printf("Number of Trials: %d\n", numTrials);
    printf("======================================================\n");
    
    // CPU - JACOBI
#ifdef RUN_CPU_FLAG
	int cpuIterations = jacobiCpuIterationCount(initX, rhs, nGrids, TOL);
    double cpuJacobiTimeTrial;
    double cpuJacobiTimeAverage;
    double cpuTotalTime = 0.0;
    float cpuJacobiResidual;
    float * solutionJacobiCpu;
    for (int iter = 0; iter < numTrials; iter++) {
		clock_t cpuJacobiStartTime = clock();
		solutionJacobiCpu = jacobiCpu(initX, rhs, nGrids, cpuIterations);
		clock_t cpuJacobiEndTime = clock();
	    cpuJacobiTimeTrial = (cpuJacobiEndTime - cpuJacobiStartTime) / (float) CLOCKS_PER_SEC;
	    cpuJacobiTimeTrial = cpuJacobiTimeTrial * (1e3); // Convert to ms
        cpuTotalTime = cpuTotalTime + cpuJacobiTimeTrial;
        printf("Completed CPU trial %d\n", iter);
    }
    cpuJacobiTimeAverage = cpuTotalTime / numTrials;
	cpuJacobiResidual = residual1DPoisson(solutionJacobiCpu, rhs, nGrids);
#endif

    // GPU - JACOBI
#ifdef RUN_GPU_FLAG
	int gpuIterations = jacobiGpuIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock);
	float gpuJacobiTimeTrial;
	float gpuJacobiTimeAverage;
    float gputotalTime = 0.0;
	float gpuJacobiResidual;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    float * solutionJacobiGpu;
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
#endif
 
    // SHARED - JACOBI
#ifdef RUN_SHARED_FLAG
	int sharedCycles = jacobiSharedIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
    float sharedJacobiTimeTrial;
    float sharedJacobiTimeAverage;
    float sharedtotalTime = 0;
    float sharedJacobiResidual;
	cudaEvent_t start_sh, stop_sh;
	cudaEventCreate(&start_sh);
	cudaEventCreate(&stop_sh);
    float * solutionJacobiShared;
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
#endif
 
    // PRINT SOLUTION - NEEDS ADJUSTING BASED ON WHICH FLAGS ARE ON
/*    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f\n", i, solutionJacobiShared[i]);
    }
*/
    
    // CPU RESULTS
#ifdef RUN_CPU_FLAG 
	printf("===============CPU============================\n");
	printf("Number of Iterations needed for Jacobi CPU: %d \n", cpuIterations);
	printf("Time needed for the Jacobi CPU: %f ms\n", cpuJacobiTimeAverage);
	printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
	std::ofstream cpuResults;
	cpuResults.open(CPU_FILE_NAME, std::ios::app);
	cpuResults << nDim << " " << residualReductionFactor << " " << numTrials << " " << cpuJacobiTimeAverage << " " << cpuIterations << " " << cpuJacobiResidual << "\n";
	cpuResults.close();
#endif
 
    // GPU RESULTS
#ifdef RUN_GPU_FLAG
	printf("===============GPU============================\n");
	printf("Number of Iterations needed for Jacobi GPU: %d \n", gpuIterations);
	printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTimeAverage);
	printf("Residual of the Jacobi GPU solution is %f\n", gpuJacobiResidual);
	std::ofstream gpuResults;
	gpuResults.open(GPU_FILE_NAME, std::ios::app);
	gpuResults << nDim << " " << threadsPerBlock << " " << residualReductionFactor << " " << numTrials << " " << gpuJacobiTimeAverage << " " << gpuIterations << " " << gpuJacobiResidual << "\n";
	gpuResults.close();
#endif

    // SHARED RESULTS
#ifdef RUN_SHARED_FLAG 
	printf("===============SHARED============================\n");
	printf("Number of Cycles needed for Jacobi Shared: %d (%d) \n", sharedCycles, threadsPerBlock/2);
	printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTimeAverage);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
	std::ofstream sharedResults;
	sharedResults.open(SHARED_FILE_NAME, std::ios::app);
	sharedResults << nDim << " " << threadsPerBlock << " " << residualReductionFactor << " " << numTrials << " " <<  sharedJacobiTimeAverage << " " << sharedCycles << " " << subIterations << " " << sharedJacobiResidual << "\n";
	sharedResults.close();
#endif

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
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
