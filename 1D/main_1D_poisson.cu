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
#include "jacobi-1D-cpu.h"
#include "jacobi-1D-gpu.h"
#include "jacobi-1D-shared.h"

#define RUN_CPU_FLAG 1
#define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1

int main(int argc, char *argv[])
{
    // INPUTS AND OUTPUT FILE NAMES
    const int nDim = 1024; //atoi(argv[1]); 
    const int threadsPerBlock = 32; //atoi(argv[2]); 
    const float TOL = 1.0; //atoi(argv[4]);
    const int OVERLAP = 0;
    const int subIterations = threadsPerBlock / 2;
    std::string CPU_FILE_NAME = "RESULTS/CPU_N1024_TOL1.txt";
    std::string GPU_FILE_NAME = "RESULTS/GPU_N1024_TOL1.txt";
    std::string SHARED_FILE_NAME = "RESULTS/SHARED_N1024_TOL1.txt";

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
    
    // CPU - JACOBI
#ifdef RUN_CPU_FLAG
	int cpuIterations = jacobiCpuIterationCount(initX, rhs, nGrids, TOL);
	clock_t cpuJacobiStartTime = clock();
	float * solutionJacobiCpu = jacobiCpu(initX, rhs, nGrids, cpuIterations);
	clock_t cpuJacobiEndTime = clock();
	double cpuJacobiTime = (cpuJacobiEndTime - cpuJacobiStartTime) / (float) CLOCKS_PER_SEC;
	cpuJacobiTime = cpuJacobiTime * (1e3); // Convert to ms
	float cpuJacobiResidual = residual1DPoisson(solutionJacobiCpu, rhs, nGrids);
#endif

    // GPU - JACOBI
#ifdef RUN_GPU_FLAG
	int gpuIterations = jacobiGpuIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float * solutionJacobiGpu = jacobiGpu(initX, rhs, nGrids, gpuIterations, threadsPerBlock);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpuJacobiTime;
	cudaEventElapsedTime(&gpuJacobiTime, start, stop);
	float gpuJacobiResidual = residual1DPoisson(solutionJacobiGpu, rhs, nGrids);
#endif
 
    // SHARED - JACOBI
#ifdef RUN_SHARED_FLAG
	int sharedCycles = jacobiSharedIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
	cudaEvent_t start_sh, stop_sh;
	cudaEventCreate(&start_sh);
	cudaEventCreate(&stop_sh);
	cudaEventRecord(start_sh, 0);
	float * solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles, threadsPerBlock, OVERLAP, subIterations);
	cudaEventRecord(stop_sh, 0);
	cudaEventSynchronize(stop_sh);
	float sharedJacobiTime;
	cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
	float sharedJacobiResidual = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
#endif
 
/*    // PRINT SOLUTION - NEEDS ADJUSTING BASED ON WHICH FLAGS ARE ON
    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f %f\n", i, solutionJacobiCpu[i], solutionJacobiGpu[i]);
    }
*/

    // PRINTOUT AND WRITE RESULTS TO FILE
   
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("======================================================\n");
    
    // CPU RESULTS
#ifdef RUN_CPU_FLAG 
	printf("===============CPU============================\n");
	printf("Number of Iterations needed for Jacobi CPU: %d \n", cpuIterations);
	printf("Time needed for the Jacobi CPU: %f ms\n", cpuJacobiTime);
	printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
	std::ofstream cpuResults;
	cpuResults.open(CPU_FILE_NAME, std::ios::app);
	cpuResults << nDim << " " << cpuIterations << " " << cpuJacobiTime << " " << cpuJacobiResidual << "\n";
	cpuResults.close();
#endif
 
    // GPU RESULTS
#ifdef RUN_GPU_FLAG
	printf("===============GPU============================\n");
	printf("Number of Iterations needed for Jacobi GPU: %d \n", gpuIterations);
	printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTime);
	printf("Residual of the Jacobi GPU solution is %f\n", gpuJacobiResidual);
	std::ofstream gpuResults;
	cpuResults.open(GPU_FILE_NAME, std::ios::app);
	cpuResults << nDim << " " << threadsPerBlock << " " << gpuIterations << " " << gpuJacobiTime << " " << gpuJacobiResidual << "\n";
	cpuResults.close();
#endif

    // SHARED RESULTS
#ifdef RUN_SHARED_FLAG 
	printf("===============SHARED============================\n");
	printf("Number of Cycles needed for Jacobi Shared: %d (%d) \n", sharedCycles, threadsPerBlock/2);
	printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTime);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
	std::ofstream sharedResults;
	cpuResults.open(SHARED_FILE_NAME, std::ios::app);
	cpuResults << nDim << " " << threadsPerBlock << " " << sharedCycles << " " << sharedJacobiTime << " " << sharedJacobiResidual << "\n";
	cpuResults.close();
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
