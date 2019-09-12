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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);   
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        float * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU Time is %f\n", elapsedTime);   

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

int jacobiGpuIterationCount(const float * initX, const float * rhs, const int nGrids, const float TOL,
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
    float residual = 100.0;
    int iIter = 0;
    float * solution = new float[nGrids];
    while (residual > TOL) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        float * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        iIter++;
        // Write solution from GPU to CPU variable
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
        residual = residual1DPoisson(solution, rhs,  nGrids);
        if (iIter % 1000 == 0) {
            printf("GPU: The residual at step %d is %f\n", iIter, residual);
        }
    }

    // Free all memory
    delete[] solution;
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    int nIters = iIter;
    return nIters;
}

