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
#include "jacobi-2D-longer-subdomain-shared.h"

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
    const int threadsPerBlock_y = 32; //4 atoi(argv[2]); 
    const int innerSubdomainLength_x = 64;
    const int innerSubdomainLength_y = 64;
    // const float TOL = 1.0; //atoi(argv[4]);
    const float residualReductionFactor = 10000.0; 
    const int trials = 1;
    const int subIterations = innerSubdomainLength_x/2;
    std::string SHARED_FILE_NAME = "OVERLAP_RESULTS/TITAN_V/SHARED.NX256.NY256.TPBX64.TPBY64.NITER32.TRUNCATED.EFFICIENT.TOLREDUCE10000.txt";
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

    // NUMBER OF OVERLAP PARAMETERS TO EXPLORE
    int numIterations_x = innerSubdomainLength_x/8; // /2
    int numIterations_y = innerSubdomainLength_y/8; // /2
    int numIterations = numIterations_x * numIterations_y;

    // DEFINE CUDA EVENTS
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);

    // NECESSARY CONTAINERS/VARIABLES
    int * sharedCycles = new int[numIterations];
    float * sharedJacobiTimeArray = new float[numIterations];
    float * residualJacobiShared = new float[numIterations];
    float * solutionJacobiShared = new float[nDofs];

    // PERFORM COMPUTATIONS FOR ALL COMBINATIONS OF OVERLAP_X AND OVERLAP_Y
    int index;
    int OVERLAP_X, OVERLAP_Y;
    float sharedJacobiTime;
    float totalTime = 0.0;
    for (int j = 0; j < numIterations_y; j++) {
        for (int i = 0; i < numIterations_x; i++) {
            OVERLAP_X = 2*i;
            OVERLAP_Y = 2*j;
            index = i + j * numIterations_x; 
            if (threadsPerBlock_x == innerSubdomainLength_x && threadsPerBlock_y == innerSubdomainLength_y) {
				sharedCycles[index] = jacobiSharedIterationCount(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations);
			}
            else {
				sharedCycles[index] = jacobiSharedLongerSubdomainIterationCount(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations, innerSubdomainLength_x, innerSubdomainLength_y);
            }
            for (int iter = 0; iter < trials; iter++) {
			    if (threadsPerBlock_x == innerSubdomainLength_x && threadsPerBlock_y == innerSubdomainLength_y) {
					cudaEventRecord(start_sh, 0);
					solutionJacobiShared = jacobiShared(initX, rhs, nxGrids, nyGrids, sharedCycles[index], threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations);
					cudaEventRecord(stop_sh, 0);
					cudaEventSynchronize(stop_sh);
					cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
					totalTime = totalTime + sharedJacobiTime;
                }
			    else {
					cudaEventRecord(start_sh, 0);
					solutionJacobiShared = jacobiSharedLongerSubdomain(initX, rhs, nxGrids, nyGrids, sharedCycles[index], threadsPerBlock_x, threadsPerBlock_y, OVERLAP_X, OVERLAP_Y, subIterations, innerSubdomainLength_x, innerSubdomainLength_y);
					cudaEventRecord(stop_sh, 0);
					cudaEventSynchronize(stop_sh);
					cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
					totalTime = totalTime + sharedJacobiTime;
                }
                printf("FINISHED ITERATION %d/%d\n", iter+1, trials);
            }
            sharedJacobiTimeArray[index] = totalTime / trials;
            residualJacobiShared[index] = residual2DPoisson(solutionJacobiShared, rhs, nxGrids, nyGrids);
            printf("FINISHED OVERLAP_X = %d/%d, OVERLAP_Y = %d/%d CASE (Nx = %d, Ny = %d, tpb_x = %d, tpb_y = %d)\n", OVERLAP_X, innerSubdomainLength_x-2, OVERLAP_Y, innerSubdomainLength_y-2, nxDim, nyDim, threadsPerBlock_x, threadsPerBlock_y);
            totalTime = 0.0;
        }
    }

    // PRINTOUT AND WRITE RESULTS TO FILE
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns in x: %d\n", nxDim);
    printf("Number of unknowns in y: %d\n", nyDim);
    printf("Threads Per Block in x and y: %d and %d\n", threadsPerBlock_x, threadsPerBlock_y);
    printf("Subdomain Length in x and y: %d and %d\n", innerSubdomainLength_x, innerSubdomainLength_y);
    printf("Residual of initial solution %f\n", initResidual);
    printf("Desired TOL of residual %f\n", TOL);
    printf("Residual reduction factor %f\n", residualReductionFactor);
    printf("======================================================\n");
    
    std::ofstream overlap_timings_sh;
    overlap_timings_sh.open(SHARED_FILE_NAME.c_str(), std::ios_base::app);
    for (int j = 0; j < numIterations_y; j++) {
        for (int i = 0; i < numIterations_x; i++) {
            OVERLAP_X = 2*i;
            OVERLAP_Y = 2*j;
            index = i + j * numIterations_x; 
			printf("================================================\n");
			printf("Number of Cycles needed for Jacobi Shared: %d (OVERLAP_X = %d, OVERLAP_Y = %d, SUBITERATIONS = %d) \n", sharedCycles[index], OVERLAP_X, OVERLAP_Y, subIterations);
			printf("Time needed for the Jacobi GPU: %f ms\n", sharedJacobiTimeArray[index]);
			printf("Residual of the Jacobi Shared solution is %f\n", residualJacobiShared[index]);
			overlap_timings_sh << OVERLAP_X << " " << OVERLAP_Y << " " << sharedCycles[index] << " " << sharedJacobiTimeArray[index] << " " << residualJacobiShared[index] << "\n";
		}
    }
    overlap_timings_sh.close();
    
    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] solutionJacobiShared;
    
    return 0;
}
