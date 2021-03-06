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
#include "jacobi-1D-longer-subdomain-shared.h"

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

    // INPUTS AND OUTPUT FILE NAMES
    const int nDim = 65536; //atoi(argv[1]);
    const int threadsPerBlock = 256; //atoi(argv[2]);
    const int innerSubdomainLength = 256;
    // const float TOL = 1.0; //atoi(argv[4]);
    const float residualReductionFactor = 10000.0;
    const int trials = 1;
    std::string FILENAME = "SUBITERATION_RESULTS/TITAN_V/N65536_TOLREDUCE10000_TPB256.txt";
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

    // COMPUTE INITIAL RESIDUAL AND SET TOLERANCE
    float initResidual = residual1DPoisson(initX, rhs, nGrids);
    const float TOL = initResidual / residualReductionFactor; //atoi(argv[4]);

    // NUMBER OF SUBITERATION AND OVERLAP PARAMETERS TO EXPLORE
    int numOverlap = innerSubdomainLength/2;
    int numSubIteration = 0;
    for (int i = innerSubdomainLength / 4; i < innerSubdomainLength * innerSubdomainLength; i = i * 2) {
        numSubIteration = numSubIteration + 1;
    }
    int numIterations = numOverlap * numSubIteration;
    
    // NECESSARY CONTAINERS
    int * sharedCycles = new int[numIterations];
    float * sharedJacobiTimeArray = new float[numIterations];
    float * residualJacobiShared = new float[numIterations];
    float * solutionJacobiShared = new float[nGrids];
    
    // DEFINE CUDA EVENTS
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    
    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns: %d\n", nDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Number of Points handled by each block: %d\n", innerSubdomainLength);
    printf("SubIteration Values to Explore: [%d %d]\n", innerSubdomainLength/4, innerSubdomainLength * innerSubdomainLength/2);
    printf("Overlap Range to Explore: [%d, %d]\n", 0, innerSubdomainLength-2);
    printf("======================================================\n");
    
    int OVERLAP, SUBITERATIONS;
    int index;
    float sharedJacobiTime;
    float totalTime = 0.0;
    // VARY NUMBER OF SUBITERATIONS
    for (int k = 0; k < numSubIteration; k++) {
        SUBITERATIONS = (innerSubdomainLength / 4) * pow(2, k);
		// VARY OVERLAP
        for (int i = 0; i < numOverlap; i++) {
			// OBTAIN NUMBER OF CYCLES TO CONVERGE FOR GIVEN COMBINATION OF OVERLAP AND SUBITERATIONS
			OVERLAP = 2*i;
            index = i + k * numOverlap;
            if (threadsPerBlock == innerSubdomainLength) {
				sharedCycles[index] = jacobiSharedIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, SUBITERATIONS);
            }
            else {
				sharedCycles[index] = jacobiSharedLongerSubdomainIterationCount(initX, rhs, nGrids, TOL, threadsPerBlock, OVERLAP, SUBITERATIONS, innerSubdomainLength);
            }
            // PERFORM TRIALS
			for (int iter = 0; iter < trials; iter++) {
				if (threadsPerBlock == innerSubdomainLength) {
					// GET FINAL SOLUTION
					cudaEventRecord(start_sh, 0);
					solutionJacobiShared = jacobiShared(initX, rhs, nGrids, sharedCycles[index], threadsPerBlock, OVERLAP, SUBITERATIONS);
					// OBTAIN FINAL TIMES REQUIRED
					cudaEventRecord(stop_sh, 0);
					cudaEventSynchronize(stop_sh);
					cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
					totalTime = totalTime + sharedJacobiTime;
                }
				else {
					// GET FINAL SOLUTION
					cudaEventRecord(start_sh, 0);
					solutionJacobiShared = jacobiSharedLongerSubdomain(initX, rhs, nGrids, sharedCycles[index], threadsPerBlock, OVERLAP, SUBITERATIONS, innerSubdomainLength);
					// OBTAIN FINAL TIMES REQUIRED
					cudaEventRecord(stop_sh, 0);
					cudaEventSynchronize(stop_sh);
					cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
					totalTime = totalTime + sharedJacobiTime;
                }
				printf("FINISHED ITERATION %d\n", iter);
			}
			sharedJacobiTimeArray[index] = totalTime / trials;
			residualJacobiShared[index] = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
            printf("Residual is %f\n", residualJacobiShared[index]);
			printf("FINISHED SUBITERATIONS %d/%d OVERLAP = %d/%d CASE (N = %d, tpb = %d)\n", SUBITERATIONS, innerSubdomainLength * innerSubdomainLength / 2, OVERLAP, innerSubdomainLength-2, nDim, threadsPerBlock);
			totalTime = 0.0;
		}    
    }
  
    // RECORD TIMINGS FOR (OVERLAP, SUBITERATIONS) IN FILE 
    std::ofstream timings_sh;
    timings_sh.open(FILENAME.c_str(), std::ios_base::app);
    for (int k = 0; k < numSubIteration; k++) {
        SUBITERATIONS = (threadsPerBlock / 4) * pow(2, k);
		for (int i = 0; i < numOverlap; i++) {
            OVERLAP = 2 * i;
            index = i + k * numOverlap;
			timings_sh << OVERLAP << " " << SUBITERATIONS << " " << sharedCycles[index] << " " << sharedJacobiTimeArray[index] << " " << residualJacobiShared[index] << "\n";
		}
    }
    timings_sh.close(); 

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solutionJacobiShared;
    
    return 0;
}
