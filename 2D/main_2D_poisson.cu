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
#include "jacobi-2D-cpu.h"
#include "jacobi-2D-gpu.h"
#include "jacobi-2D-shared.h"

// #define RUN_CPU_FLAG 1
// #define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1

int main(int argc, char *argv[])
{

    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "Quadro K420";
    setGPU(gpuToUse);

    // INPUTS AND OUTPUT FILE NAMES
    const int nxDim = 256; //atoi(argv[1]); 
    const int nyDim = 256; //atoi(argv[1]); 
    const int threadsPerBlock_x = 32; //atoi(argv[2]); 
    const int threadsPerBlock_y = 32; //atoi(argv[2]); 
    // const float TOL = 1.0; //atoi(argv[4]);
    const float residualReductionFactor = 10000.0; 
    const int OVERLAP_X = 0;
    const int OVERLAP_Y = 0;
    const int subIterations = std::min(threadsPerBlock_x, threadsPerBlock_y) / 2;
    std::string CPU_FILE_NAME = "RESULTS/CPU_N1024_TOL1.txt";
    std::string GPU_FILE_NAME = "RESULTS/GPU_N1024_TOL1.txt";
    std::string SHARED_FILE_NAME = "RESULTS/SHARED_N1024_TOL1.txt";
    /////////////////////////////////////////////////////////////////////////

    // INITIALIZE ARRAYS
    int nxGrids = nxDim + 2;
    int nyGrids = nyDim + 2;
    int nDofs = nxGrids * nyGrids;
    float * initX = new float[nDofs];
    float * rhs = new float[nDofs];
    
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
    
    // COMPUTE INITIAL RESIDUAL AND SET TOLERANCE
    float initResidual = residual2DPoisson(initX, rhs, nxGrids, nyGrids);
    const float TOL = initResidual / residualReductionFactor; //atoi(argv[4]);

    // CPU - JACOBI
#ifdef RUN_CPU_FLAG
	int cpuIterations = jacobiCpuIterationCount(initX, rhs, nxGrids, nyGrids, TOL);
	clock_t cpuJacobiStartTime = clock();
	float * solutionJacobiCpu = jacobiCpu(initX, rhs, nxGrids, nyGrids, cpuIterations);
	clock_t cpuJacobiEndTime = clock();
	double cpuJacobiTime = (cpuJacobiEndTime - cpuJacobiStartTime) / (float) CLOCKS_PER_SEC;
	cpuJacobiTime = cpuJacobiTime * (1e3); // Convert to ms
	float cpuJacobiResidual = residual2DPoisson(solutionJacobiCpu, rhs, nxGrids, nyGrids); 
#endif

    // GPU - JACOBI
#ifdef RUN_GPU_FLAG
	int gpuIterations = jacobiGpuIterationCount(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float * solutionJacobiGpu = jacobiGpu(initX, rhs, nxGrids, nyGrids, gpuIterations, threadsPerBlock_x, threadsPerBlock_y);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpuJacobiTime;
	cudaEventElapsedTime(&gpuJacobiTime, start, stop);
	float gpuJacobiResidual = residual2DPoisson(solutionJacobiGpu, rhs, nxGrids, nyGrids);
#endif

    // SHARED - JACOBI
#ifdef RUN_SHARED_FLAG
    int sharedCycles = jacobiSharedIterationCount(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations);
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    cudaEventRecord(start_sh, 0);
    float * solutionJacobiShared = jacobiShared(initX, rhs, nxGrids, nyGrids, sharedCycles, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations);
    cudaEventRecord(stop_sh, 0);
    cudaEventSynchronize(stop_sh);
    float sharedJacobiTime;
    cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
    float sharedJacobiResidual = residual2DPoisson(solutionJacobiShared, rhs, nxGrids, nyGrids);
#endif
 
    // PRINT SOLUTION - NEEDS ADJUSTING BASED ON WHICH FLAGS ARE ON
/*    for (int i = 0; i < nDofs; i++) {
        printf("Grid %d = %f %f\n", i, solutionJacobiCpu[i], solutionJacobiGpu[i]);
    }
*/

    // PRINTOUT AND WRITE RESULTS TO FILE
   
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns in x: %d\n", nxDim);
    printf("Number of unknowns in y: %d\n", nyDim);
    printf("Threads Per Block in x: %d\n", threadsPerBlock_x);
    printf("Threads Per Block in y: %d\n", threadsPerBlock_y);
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
	cpuResults << nxDim << " " << nyDim << " " << cpuIterations << " " << cpuJacobiTime << " " << cpuJacobiResidual << "\n";
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
	gpuResults << nxDim << " " << nyDim << " " << threadsPerBlock_x << " " << threadsPerBlock_y << " " << gpuIterations << " " << gpuJacobiTime << " " << gpuJacobiResidual << "\n";
	gpuResults.close();
#endif

    // SHARED RESULTS
#ifdef RUN_SHARED_FLAG
    printf("===============SHARED============================\n");
    printf("Number of Cycles needed for Jacobi Shared: %d (OVERLAP_X = %d, OVERLAP_Y = %d, SUBITERATIONS = %d) \n", sharedCycles, OVERLAP_X, OVERLAP_Y, subIterations);
    printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTime);
    printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
    std::ofstream sharedResults;
    sharedResults.open(SHARED_FILE_NAME.c_str(), std::ios::app);
    sharedResults << nxDim << " " << nyDim << " " << threadsPerBlock_x << " " << threadsPerBlock_y << " " << sharedCycles << " " << sharedJacobiTime << " " << sharedJacobiResidual << "\n";
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
