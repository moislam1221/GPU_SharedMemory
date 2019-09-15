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

int main(int argc, char *argv[])
{
    // INPUTS
    const int nDim = 1024; //atoi(argv[1]); 
    const int threadsPerBlock = 128; //atoi(argv[2]); 
    const float TOL = 1.0; //atoi(argv[4]);
    const int OVERLAP = 0;

    // INITIALIZE ARRAYS
    int nGrids = nDim + 2;
    float * initX = new float[nGrids];
    float * rhs = new float[nGrids];
    
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

    // OBTAIN NUMBER OF ITERATIONS NECESSARY TO ACHIEVE TOLERANCE FOR EACH METHOD
    int cpuIterations = jacobiCpuIterationCount(initX, rhs, nGrids, TOL);
    int gpuIterations = jacobiGpuIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock);
    int sharedCycles = jacobiSharedIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP);
    
    // CPU - JACOBI
    clock_t cpuJacobiStartTime = clock();
    float * solutionJacobiCpu = jacobiCpu(initX, rhs, nGrids, cpuIterations);
    clock_t cpuJacobiEndTime = clock();
    double cpuJacobiTime = (cpuJacobiEndTime - cpuJacobiStartTime) / (float) CLOCKS_PER_SEC;
    cpuJacobiTime = cpuJacobiTime * (1e3); // Convert to ms

    // GPU - JACOBI
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float * solutionJacobiGpu = jacobiGpu(initX, rhs, nGrids, gpuIterations, threadsPerBlock);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuJacobiTime;
    cudaEventElapsedTime(&gpuJacobiTime, start, stop);
    
    // SHARED - JACOBI
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    cudaEventRecord(start_sh, 0);
    float * solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles, threadsPerBlock, OVERLAP);
    cudaEventRecord(stop_sh, 0);
    cudaEventSynchronize(stop_sh);
    float sharedJacobiTime;
    cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
    
    // PRINT SOLUTION
    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f %f\n", i, solutionJacobiCpu[i], solutionJacobiGpu[i]);
    }

    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("======================================================\n");
    
    // Print out number of iterations needed for each method
    printf("Number of Iterations needed for Jacobi CPU: %d \n", cpuIterations);
    printf("Number of Iterations needed for Jacobi GPU: %d \n", gpuIterations);
    printf("Number of Cycles needed for Jacobi Shared: %d (%d) \n", sharedCycles, threadsPerBlock/2);
    
    // Print out time for cpu, classic gpu, and swept gpu approaches
    printf("Time needed for the Jacobi CPU: %f ms\n", cpuJacobiTime);
    printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTime);
    printf("Time needed for the Jacobi GPU: %f ms\n", sharedJacobiTime);
    printf("======================================================\n");

    // Compute the residual of the resulting solution (|b-Ax|)
    float residualJacobiCpu = residual1DPoisson(solutionJacobiCpu, rhs, nGrids);
    float residualJacobiGpu = residual1DPoisson(solutionJacobiGpu, rhs, nGrids);
    float residualJacobiShared = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
    printf("Residual of the Jacobi CPU solution is %f\n", residualJacobiCpu);
    printf("Residual of the Jacobi GPU solution is %f\n", residualJacobiGpu);
    printf("Residual of the Jacobi Shared solution is %f\n", residualJacobiShared);

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solutionJacobiCpu;
    delete[] solutionJacobiGpu;
    delete[] solutionJacobiShared;
    
    return 0;
}
