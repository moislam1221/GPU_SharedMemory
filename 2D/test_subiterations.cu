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
#include "jacobi-2D-shared.h"

int main(int argc, char *argv[])
{
    // INPUTS ///////////////////////////////////////////////////////////////
    // SET CUDA DEVICE TO USE (IMPORTANT FOR ENDEAVOUR WHICH HAS 2!)
    // NAVIER-STOKES GPUs: "Quadro K420"
    // ENDEAVOUR GPUs: "TITAN V" OR "GeForce GTX 1080 Ti"
    std::string gpuToUse = "TITAN V";
    setGPU(gpuToUse);

    // INPUTS AND OUTPUT FILE NAMES
    const int nxDim = 256; //16 atoi(argv[1]); 
    const int nyDim = 256; //16 atoi(argv[1]); 
    const int threadsPerBlock_x = 32; //4 atoi(argv[2]); 
    const int threadsPerBlock_y = 16; //4 atoi(argv[2]); 
    // const float TOL = 1.0; //atoi(argv[4]);
    const float residualReductionFactor = 10000.0; 
    const int trials = 1;
    std::string SHARED_FILE_NAME = "SUBITERATION_RESULTS/TITAN_V/SHARED.NX256.NY256.TPBX32.TPBY16.TOLREDUCE10000.TRUNCATED.txt";
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
				initX[dof] = 0.0f; //(float)dof;
			}
			else {
				initX[dof] = 1.0f; //(float)dof; 
			}
			rhs[dof] = 1.0f;
        }
    }
    
    // COMPUTE INITIAL RESIDUAL AND SET TOLERANCE
    float initResidual = residual2DPoisson(initX, rhs, nxGrids, nyGrids);
    const float TOL = initResidual / residualReductionFactor; //atoi(argv[4]);

    // NUMBER OF SUBITERATION OVERLAP PARAMETERS TO EXPLORE
    int numOverlap_x = threadsPerBlock_x/4;
    int numOverlap_y = threadsPerBlock_y/4;
    int maxDim = max(threadsPerBlock_x, threadsPerBlock_y);
    int numSubIteration = 0;
    for (int i = maxDim / 4; i <= maxDim * maxDim / 2; i = i*2) { 
        numSubIteration = numSubIteration + 1;
    }
    int numIterations = numOverlap_x * numOverlap_y * numSubIteration;

    // DEFINE CUDA EVENTS
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);

    // NECESSARY CONTAINERS/VARIABLES
    int * sharedCycles = new int[numIterations];
    float * sharedJacobiTimeArray = new float[numIterations];
    float * residualJacobiShared = new float[numIterations];
    float * solutionJacobiShared = new float[nDofs];
    
    // PRINTOUT AND WRITE RESULTS TO FILE
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns in x: %d\n", nxDim);
    printf("Number of unknowns in y: %d\n", nyDim);
    printf("Threads Per Block in x and y: %d and %d\n", threadsPerBlock_x, threadsPerBlock_y);
    printf("Overlap Range to Explore in x: [%d, %d]\n", 0, threadsPerBlock_x-2);
    printf("Overlap Range to Explore in y: [%d, %d]\n", 0, threadsPerBlock_y-2);
    printf("SubIteration Values to Explore: [%d %d]\n", maxDim/4, maxDim * maxDim / 2);
    printf("Residual of initial solution %f\n", initResidual);
    printf("Desired TOL of residual %f\n", TOL);
    printf("Residual reduction factor %f\n", residualReductionFactor);
    printf("======================================================\n");

    // PERFORM COMPUTATIONS FOR ALL COMBINATIONS OF OVERLAP_X AND OVERLAP_Y
    int OVERLAP_X, OVERLAP_Y, SUBITERATIONS;
    int index;
    float sharedJacobiTime;
    float totalTime = 0.0;
    // VARY NUMBER OF SUBITERATIONS
    for (int k = 0; k < numSubIteration; k++) {
        SUBITERATIONS = (maxDim / 4) * pow(2, k);
        // VARY OVERLAP IN Y
		for (int j = 0; j < numOverlap_y; j++) {
            // VARY OVERLAP IN X
			for (int i = 0; i < numOverlap_x; i++) {
				OVERLAP_X = 2*i;
				OVERLAP_Y = 2*j;
				index = i + j * numOverlap_x + k * numOverlap_x * numOverlap_y; 
				sharedCycles[index] = jacobiSharedIterationCount(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, SUBITERATIONS);
				for (int iter = 0; iter < trials; iter++) {
					cudaEventRecord(start_sh, 0);
					solutionJacobiShared = jacobiShared(initX, rhs, nxGrids, nyGrids, sharedCycles[index], threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, SUBITERATIONS);
					cudaEventRecord(stop_sh, 0);
					cudaEventSynchronize(stop_sh);
					cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
					totalTime = totalTime + sharedJacobiTime;
					printf("FINISHED ITERATION %d/%d\n", iter+1, trials);
				}
				sharedJacobiTimeArray[index] = totalTime / trials;
				residualJacobiShared[index] = residual2DPoisson(solutionJacobiShared, rhs, nxGrids, nyGrids);
				printf("FINISHED SUBITERATIONS %d/%d OVERLAP_X = %d/%d, OVERLAP_Y = %d/%d, CASE (Nx = %d, Ny = %d, tpb_x = %d, tpb_y = %d)\n", SUBITERATIONS, maxDim * maxDim / 2, OVERLAP_X, threadsPerBlock_x-2, OVERLAP_Y, threadsPerBlock_y-2, nxDim, nyDim, threadsPerBlock_x, threadsPerBlock_y);
				totalTime = 0.0;
			}
		}
    }

    std::ofstream subiteration_timings_sh;
    subiteration_timings_sh.open(SHARED_FILE_NAME.c_str(), std::ios_base::app);
    for (int k = 0; k < numSubIteration; k++) {
        SUBITERATIONS = (maxDim / 4) * pow(2, k);
		for (int j = 0; j < numOverlap_y; j++) {
			for (int i = 0; i < numOverlap_x; i++) {
				OVERLAP_X = 2*i;
				OVERLAP_Y = 2*j;
				index = i + j * numOverlap_x + k * numOverlap_x * numOverlap_y; 
				subiteration_timings_sh << OVERLAP_X << " " << OVERLAP_Y << " " << SUBITERATIONS << " " << sharedCycles[index] << " " << sharedJacobiTimeArray[index] << " " << residualJacobiShared[index] << "\n";
			}
		}
    }
    subiteration_timings_sh.close();
    
    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solutionJacobiShared;
    
    return 0;
}
