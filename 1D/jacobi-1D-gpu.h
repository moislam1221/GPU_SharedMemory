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

__global__
void _jacobiGpuClassicIteration(float * x1, const float * x0, const float * rhs, const int nGrids, const float dx)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid > 0 && iGrid < nGrids - 1) {
        float leftX = x0[iGrid - 1];
        float rightX = x0[iGrid + 1];
        x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
    }
    __syncthreads();
}

float * jacobiGpu(const float * initX, const float * rhs, const int nGrids, const int nIters,
                  const int threadsPerBlock)
{
    // Compute dx for use in jacobi1DPoisson
    float dx = 1.0 / (nGrids - 1);
    
    // Allocate memory in the CPU for all inputs and solutions
    float * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);

    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        float * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiGpuIterationCount(const float * initX, const float * solution_exact, const float * rhs, const int nGrids, const float TOL, const int threadsPerBlock)
{
    // Compute dx for use in jacobi1DPoisson
    float solution_error = 1000000000000.0;
    float dx = 1.0 / (nGrids - 1);
    
    // Allocate memory in the CPU for all inputs and solutions
    float * x0Gpu, * x1Gpu, * rhsGpu, * solutionErrorGpu, * solution_exactGpu; // * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    // cudaMalloc(&residualGpu, sizeof(float) * nGrids);
    cudaMalloc(&solutionErrorGpu, sizeof(float) * nGrids);
    cudaMalloc(&solution_exactGpu, sizeof(float) * nGrids);

    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    // cudaMemcpy(residualGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(solutionErrorGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(solution_exactGpu, solution_exact, sizeof(float) * nGrids, cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);
    float residual = 1000000000000.0;
    int iIter = 0;
    float * solution = new float[nGrids];
    // float * residualCpu = new float[nGrids];
    float * solutionErrorCpu = new float[nGrids];
    // while (residual > TOL) {
	while (solution_error > TOL) {
    // Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        {
            float * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
        iIter++;
        // RESIDUAL CALCULATION
        // Write solution from GPU to CPU variable
//		cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
//		residual = residual1DPoisson(solution, rhs,  nGrids);
/*        residual1DPoissonGPU <<<nBlocks, threadsPerBlock>>> (residualGpu, x0Gpu, rhsGpu, nGrids);
        cudaDeviceSynchronize();
        cudaMemcpy(residualCpu, residualGpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
        residual = 0.0;
        for (int j = 0; j < nGrids; j++) {
            residual = residual + residualCpu[j];
        }
        residual = sqrt(residual);
        if (iIter % 1000 == 0) {
			printf("GPU: The residual at step %d is %f\n", iIter, residual);
        }
*/

//      ERROR CALCULATION
//		cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
//		solution_error = solutionError1DPoisson(solution, solution_exact, nGrids);
        solutionError1DPoissonGPU <<<nBlocks, threadsPerBlock>>> (solutionErrorGpu, x0Gpu, solution_exactGpu, nGrids);
        cudaDeviceSynchronize();
        cudaMemcpy(solutionErrorCpu, solutionErrorGpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
        solution_error = 0.0;
        for (int j = 0; j < nGrids; j++) {
             solution_error = solution_error + solutionErrorCpu[j];
        }
        solution_error = sqrt(solution_error);
        if (iIter % 1000 == 0) {
			printf("GPU: The solution error at step %d is %f\n", iIter, solution_error);
        }
    }

    // Free all memory
    // CPU
    delete[] solution;
    // delete[] solution_error;
    // delete[] residualCpu;
    delete[] solutionErrorCpu;
    // GPU
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(solutionErrorGpu);

    int nIters = iIter;
    return nIters;
}

