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
#include "jacobi-1D-shared.h"

// IDEA: For N = 1024, create a plot of convergence time as the overlap points increase

int main(int argc, char *argv[])
{
    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "Quadro K420";
    setGPU(gpuToUse);

    // INPUTS AND OUTPUT FILE NAMES
    const int nDim = 1024; //atoi(argv[1]); 
    const int threadsPerBlock = 32; //atoi(argv[2]); 
    const float TOL = 1.0; //atoi(argv[4]);
    const int nIters = 2; // 20
    const int subIterations = threadsPerBlock/2; //atoi(argv[2]); 
    std::string FILENAME = "OVERLAP_RESULTS/N1024_TOL1_TPB32_DUMMY.txt";
    /////////////////////////////////////////////////////////////////////////
 
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

    // NUMBER OF OVERLAP PARAMETERS TO EXPLORE
    int numIterations = (threadsPerBlock-2)/2;
    
    // NECESSARY CONTAINERS
    int * sharedCycles = new int[subIterations];
    float * sharedJacobiTimeArray = new float[numIterations];
    float * residualJacobiShared = new float[numIterations];
    float * solutionJacobiShared = new float[nGrids];
    
    // DEFINE CUDA EVENTS
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    
    int OVERLAP;
    float sharedJacobiTime;
    float totalTime = 0.0;
    for (int i = 0; i <= numIterations; i++) {
        // OBTAIN NUMBER OF CYCLES TO CONVERGE FOR GIVEN OVERLAP
        OVERLAP = 2*i;
        sharedCycles[i] = jacobiSharedIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
        for (int iter = 0; iter < nIters; iter++) {
            // GET FINAL SOLUTION
			cudaEventRecord(start_sh, 0);
			solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles[i], threadsPerBlock, OVERLAP, subIterations);
			// OBTAIN FINAL TIMES REQUIRED
			cudaEventRecord(stop_sh, 0);
			cudaEventSynchronize(stop_sh);
			cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
			totalTime = totalTime + sharedJacobiTime;
            printf("FINISHED ITERATION %d\n", iter);
        }
        sharedJacobiTimeArray[i] = totalTime / nIters;
        residualJacobiShared[i] = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
        printf("Residual is %f\n", residualJacobiShared[i]);
        printf("FINISHED OVERLAP = %d/%d CASE (N = %d, tpb = %d)\n", OVERLAP, threadsPerBlock-2, nDim, threadsPerBlock);
        totalTime = 0.0;
    }    

    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("======================================================\n");
   
    std::ofstream timings_sh;
    timings_sh.open(FILENAME.c_str(), std::ios_base::app);
    for (int i = 0; i <= numIterations; i++) {
        int OVERLAP = 2 * i;
        printf("================================================\n");
        printf("Number of Cycles needed for Jacobi Shared for OVERLAP = %d: %d (%d) \n", OVERLAP, sharedCycles[i], threadsPerBlock/2);
        printf("Time needed for the Jacobi GPU: %f ms\n", sharedJacobiTimeArray[i]);
        printf("Residual of the Jacobi Shared solution is %f\n", residualJacobiShared[i]);
        timings_sh << OVERLAP << " " <<  sharedCycles[i] << " " << sharedJacobiTimeArray[i] << " " << residualJacobiShared[i] << "\n";
    }
    timings_sh.close(); 

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solutionJacobiShared;
    
    return 0;
}
