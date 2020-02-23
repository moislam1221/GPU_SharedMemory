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

#define GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST 1

__global__
void _jacobiGpuClassicIteration(double * x1, const double * x0, const double * rhs, const int nGrids, const double dx)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid > 0 && iGrid < nGrids - 1) {
        double leftX = x0[iGrid - 1];
        double rightX = x0[iGrid + 1];
        x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
    }
    __syncthreads();
}

double * jacobiGpu(const double * initX, const double * rhs, const int nGrids, const int nIters,
                  const int threadsPerBlock)
{
    // Compute dx for use in jacobi1DPoisson
    double dx = 1.0 / (nGrids - 1);
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);

    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    double * solution = new double[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiGpuIterationCountResidual(const double * initX, const double * rhs, const int nGrids, const double TOL, const int threadsPerBlock)
{
    // Compute dx for use in jacobi1DPoisson
    double dx = 1.0 / (nGrids - 1);
 
    // Initial residual
    double residual = 1000000000000.0;
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    cudaMalloc(&residualGpu, sizeof(double) * nGrids);

    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(residualGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);
    int iIter = 0;
    double * residualCpu = new double[nGrids];
    while (residual > TOL) {
        // Jacobi iteration on the GPU
        //for (int i = 0; i < nGrids; i++) {
        //printf("x0Gpu[%d] = %f\n", i, x0Gpu[i]);
        //}
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess)
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        {
            double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
        //for (int i = 0; i < nGrids; i++) {
        //printf("x0Gpu[%d] = %f\n", i, x0Gpu[i]);
        //}
        iIter++;
        // RESIDUAL CALCULATION
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
        residual1DPoissonGPU <<<nBlocks, threadsPerBlock>>> (residualGpu, x0Gpu, rhsGpu, nGrids);
        cudaDeviceSynchronize();
        cudaMemcpy(residualCpu, residualGpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
        residual = 0.0;
        for (int j = 0; j < nGrids; j++) {
            residual = residual + residualCpu[j];
        }
        residual = sqrt(residual);
#else
        // Write solution from GPU to CPU variable
		cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
		residual = residual1DPoisson(solution, rhs, nGrids);
#endif
        // Print out the residual
        if (iIter % 1 == 0) {
			printf("GPU: The residual at step %d is %f\n", iIter, residual);
        }
	}

    // Free all memory
    // CPU
    delete[] residualCpu;
    // GPU
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(residualGpu);

    int nIters = iIter;
    return nIters;
}

int jacobiGpuIterationCountSolutionError(const double * initX, const double * rhs, const int nGrids, const double TOL, const int threadsPerBlock, const double * solution_exact)
{
    // Compute dx for use in jacobi1DPoisson
    double dx = 1.0 / (nGrids - 1);
    
    // Initial solution error
    double solution_error = 1000000000000.0;
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu, * solutionErrorGpu, * solution_exactGpu; 
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    cudaMalloc(&solutionErrorGpu, sizeof(double) * nGrids);
    cudaMalloc(&solution_exactGpu, sizeof(double) * nGrids);

    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(solutionErrorGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(solution_exactGpu, solution_exact, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);
    int iIter = 0;
    double * solutionErrorCpu = new double[nGrids];
	while (solution_error > TOL) {
        // Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx); 
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess)
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        {
            double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
        iIter++;
//      ERROR CALCULATION
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
        solutionError1DPoissonGPU <<<nBlocks, threadsPerBlock>>> (solutionErrorGpu, x0Gpu, solution_exactGpu, nGrids);
        cudaDeviceSynchronize();
        cudaMemcpy(solutionErrorCpu, solutionErrorGpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
        solution_error = 0.0;
        for (int j = 0; j < nGrids; j++) {
             solution_error = solution_error + solutionErrorCpu[j];
        }
        solution_error = sqrt(solution_error);
#else        
		cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
		solution_error = solutionError1DPoisson(solution, solution_exact, nGrids);
#endif
        // Print out the solution error
        if (iIter % 1000 == 0) {
			printf("GPU: The solution error at step %d is %f\n", iIter, solution_error);
        }
    }

    // Free all memory
    // CPU
    delete[] solutionErrorCpu;
    
    // GPU
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(solutionErrorGpu);
    cudaFree(solution_exactGpu);

    int nIters = iIter;
    return nIters;
}

