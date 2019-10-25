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
#include "jacobi-1D-longer-subdomain-shared.h"

// #define RUN_CPU_FLAG 1
// #define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1
#define RUN_SHARED_LONGER_SUBDOMAIN_FLAG 1

int main(int argc, char *argv[])
{
    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "Quadro K420"; 
    setGPU(gpuToUse);

    // INPUTS AND OUTPUT FILE NAMES
    const int nDim = 65536; //atoi(argv[1]); 
    const int threadsPerBlock = 32; //atoi(argv[2]); 
    const int innerSubdomainLength = 32;
    const float residualReductionFactor = 1000.0; //atoi(argv[4]);
    const int OVERLAP = 0;
    const int subIterations = threadsPerBlock / 2;
    std::string CPU_FILE_NAME = "RESULTS/CPU_N65536_TOLREDUCE10000.txt";
    std::string GPU_FILE_NAME = "RESULTS/GPU_N65536_TOLREDUCE10000.txt";
    std::string SHARED_FILE_NAME = "RESULTS/SHARED_N65536_TOLREDUCE10000.txt";
    std::string SHARED_LONGER_SUBDOMAIN_FILE_NAME = "RESULTS/SHARED_LONGER_SUBDOMAIN_N65536_TOLREDUCE10000.txt";
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
 
    // SHARED WITH LONGER SUBDOMAIN - JACOBI
#ifdef RUN_SHARED_LONGER_SUBDOMAIN_FLAG
	int sharedCyclesLong = jacobiSharedLongerSubdomainIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations, innerSubdomainLength);
	cudaEvent_t start_sh_long, stop_sh_long;
	cudaEventCreate(&start_sh_long);
	cudaEventCreate(&stop_sh_long);
	cudaEventRecord(start_sh_long, 0);
	float * solutionJacobiSharedLong = jacobiSharedLongerSubdomain(initX, rhs, nGrids, sharedCycles, threadsPerBlock, OVERLAP, subIterations, innerSubdomainLength);
	cudaEventRecord(stop_sh_long, 0);
	cudaEventSynchronize(stop_sh_long);
	float sharedJacobiTimeLong;
	cudaEventElapsedTime(&sharedJacobiTimeLong, start_sh_long, stop_sh_long);
	float sharedJacobiResidualLong = residual1DPoisson(solutionJacobiSharedLong, rhs, nGrids);
#endif
/*    // PRINT SOLUTION - NEEDS ADJUSTING BASED ON WHICH FLAGS ARE ON
    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f %f\n", i, solutionJacobiCpu[i], solutionJacobiGpu[i]);
    }
*/

    // PRINTOUT AND WRITE RESULTS TO FILE
   
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Residual of initial solution %f\n", initResidual);
    printf("Desired TOL of residual %f\n", TOL);
    printf("Residual reduction factor %f\n", residualReductionFactor);
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
	gpuResults.open(GPU_FILE_NAME, std::ios::app);
	gpuResults << nDim << " " << threadsPerBlock << " " << gpuIterations << " " << gpuJacobiTime << " " << gpuJacobiResidual << "\n";
	gpuResults.close();
#endif

    // SHARED RESULTS
#ifdef RUN_SHARED_FLAG 
	printf("===============SHARED============================\n");
	printf("Number of Cycles needed for Jacobi Shared: %d (OVERLAP = %d, SUBITERATIONS = %d) \n", sharedCycles, OVERLAP, subIterations);
    printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTime);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
	std::ofstream sharedResults;
	sharedResults.open(SHARED_FILE_NAME, std::ios::app);
	sharedResults << nDim << " " << threadsPerBlock << " " << sharedCycles << " " << sharedJacobiTime << " " << sharedJacobiResidual << "\n";
	sharedResults.close();
#endif

    // SHARED RESULTS LONGER SUBDOMAIN
#ifdef RUN_SHARED_LONGER_SUBDOMAIN_FLAG 
	printf("===============SHARED (LONG)============================\n");
    printf("Length of Subdomain: %d with Threads Per Block: %d\n", innerSubdomainLength, threadsPerBlock);
	printf("Number of Cycles needed for Jacobi Shared Longer Subdomain: %d (OVERLAP = %d, SUBITERATIONS = %d) \n", sharedCyclesLong, OVERLAP, subIterations);
    printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTimeLong);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidualLong);
	std::ofstream sharedResultsLong;
	sharedResultsLong.open(SHARED_FILE_NAME, std::ios::app);
	sharedResultsLong << nDim << " " << threadsPerBlock << " " << sharedCyclesLong << " " << sharedJacobiTimeLong << " " << sharedJacobiResidualLong << "\n";
	sharedResultsLong.close();
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
#ifdef RUN_SHARED_LONGER_SUBDOMAIN_FLAG
    delete[] solutionJacobiSharedLong;
#endif
    
    return 0;
}
