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
    const int nxDim = 4; //atoi(argv[1]); 
    const int nyDim = 4; //atoi(argv[1]); 
    const int threadsPerBlock = 2; //atoi(argv[2]); 
    // const float TOL = 1.0; //atoi(argv[4]);
    const float residualReductionFactor = 1000.0; 
    const int OVERLAP = 0;
    const int subIterations = 2; //threadsPerBlock / 2;
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
				initX[dof] = (float)dof;
			}
			else {
				initX[dof] = (float)dof; 
			}
			rhs[dof] = 1.0f;
        }
    }
    
    // COMPUTE INITIAL RESIDUAL AND SET TOLERANCE
    float initResidual = residual2DPoisson(initX, rhs, nxGrids, nyGrids);
    const float TOL = initResidual / residualReductionFactor; //atoi(argv[4]);

    // SHARED - JACOBI
#ifdef RUN_SHARED_FLAG
    int sharedCycles = 1; //jacobiSharedIterationCount(initX, rhs, nxGrids, nyGrids, TOL, threadsPerBlock, OVERLAP, subIterations);
    cudaEvent_t start_sh, stop_sh;
    cudaEventCreate(&start_sh);
    cudaEventCreate(&stop_sh);
    cudaEventRecord(start_sh, 0);
    printf("Here\n");
    float * solutionJacobiShared = jacobiShared(initX, rhs, nxGrids, nyGrids, 1, threadsPerBlock, OVERLAP, subIterations);
    printf("Now Here\n");
    cudaEventRecord(stop_sh, 0);
    cudaEventSynchronize(stop_sh);
    float sharedJacobiTime;
    cudaEventElapsedTime(&sharedJacobiTime, start_sh, stop_sh);
    float sharedJacobiResidual = residual2DPoisson(solutionJacobiShared, rhs, nxGrids, nyGrids);
#endif
 
    // PRINT SOLUTION - NEEDS ADJUSTING BASED ON WHICH FLAGS ARE ON
    for (int i = 0; i < nDofs; i++) {
        printf("Grid %d = %f\n", i, solutionJacobiShared[i]);
    }

    // PRINTOUT AND WRITE RESULTS TO FILE
   
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("GPU Name: %s\n", gpuToUse.c_str());
    printf("Number of unknowns in x: %d\n", nxDim);
    printf("Number of unknowns in y: %d\n", nyDim);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Residual of initial solution %f\n", initResidual);
    printf("Desired TOL of residual %f\n", TOL);
    printf("Residual reduction factor %f\n", residualReductionFactor);
    printf("======================================================\n");
    
    // SHARED RESULTS
#ifdef RUN_SHARED_FLAG
    printf("===============SHARED============================\n");
    printf("Number of Cycles needed for Jacobi Shared: %d (OVERLAP = %d, SUBITERATIONS = %d) \n", sharedCycles, OVERLAP, subIterations);
    printf("Time needed for the Jacobi Shared: %f ms\n", sharedJacobiTime);
    printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
    std::ofstream sharedResults;
    sharedResults.open(SHARED_FILE_NAME, std::ios::app);
    sharedResults << nxDim << " " << nyDim << " " << threadsPerBlock << " " << sharedCycles << " " << sharedJacobiTime << " " << sharedJacobiResidual << "\n";
    sharedResults.close();
#endif

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
#ifdef RUN_SHARED_FLAG 
    delete[] solutionJacobiShared;
#endif
    
    return 0;
}
