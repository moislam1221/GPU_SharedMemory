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
void _jacobiGpuClassicIteration(float * x1, const float * x0, const float * rhs, const int nxGrids, const int nyGrids, const float dx, const float dy)
{
    int ixGrid = blockIdx.x * blockDim.x + threadIdx.x;
    int iyGrid = blockIdx.y * blockDim.y + threadIdx.y;
    int dof = iyGrid * nxGrids + ixGrid;
    int nDofs = nxGrids * nyGrids;
    if (dof < nDofs) {
		if ((ixGrid > 0) && (ixGrid < nxGrids - 1) && (iyGrid > 0) && (iyGrid < nyGrids - 1)) {
			float leftX = x0[dof - 1];
			float rightX = x0[dof + 1];
			float topX = x0[dof + nxGrids];
			float bottomX = x0[dof - nxGrids];
			x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
		}
    }
    __syncthreads();
}

float * jacobiGpu(const float * initX, const float * rhs, const int nxGrids, const int nyGrids,
                  const int nIters, const int threadsPerBlock)
{
    // Compute dx, dy for use in jacobi2DPoisson
    float dx = 1.0 / (nxGrids - 1);
    float dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for all inputs and solutions
    float * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Establish 2D grid and block structures
    dim3 block(threadsPerBlock, threadsPerBlock);
    const int nxBlocks = (int)ceil(nxGrids / (float)threadsPerBlock);
    const int nyBlocks = (int)ceil(nyGrids / (float)threadsPerBlock);
    dim3 grid(nxBlocks, nyBlocks);

    float * solution = new float[nDofs];
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, dx, dy); 
        float * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    // float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiGpuIterationCount(const float * initX, const float * rhs, const int nxGrids, const int nyGrids, 
                            const float TOL, const int threadsPerBlock)
{
    // Compute dx, dy for use in jacobi2DPoisson
    float dx = 1.0 / (nxGrids - 1);
    float dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for all inputs and solutions
    float * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Establish 2D grid and block structures
    dim3 block(threadsPerBlock, threadsPerBlock);
    const int nxBlocks = (int)ceil(nxGrids / (float)threadsPerBlock);
    const int nyBlocks = (int)ceil(nyGrids / (float)threadsPerBlock);
    dim3 grid(nxBlocks, nyBlocks);

    // Run the classic iteration for prescribed number of iterations
    float residual = 1000000000000.0;
    int iIter = 0;
    float * solution = new float[nDofs];
    while (residual > TOL) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, dx, dy); 
        float * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        iIter++;
        // Write solution from GPU to CPU variable
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs, cudaMemcpyDeviceToHost);
        residual = residual2DPoisson(solution, rhs, nxGrids, nyGrids);
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

