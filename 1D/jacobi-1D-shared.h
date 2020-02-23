#define GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST 1

__device__
void __jacobiUpdateKernel(const int nGrids, const int nSub, const int OVERLAP, const int subIterations)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + nSub, * x2 = sharedMemory + 2 * nSub;

    const double dx = 1.0 / (nGrids - 1);

    for (int k = 0; k < subIterations; k++) {
        int i = threadIdx.x + 1;
        int I = blockIdx.x * (blockDim.x - OVERLAP)+ i;
        if (I < nGrids-1) {
            double leftX = x0[i-1];
            double rightX = x0[i+1];
            double rhs = x2[i];
            x1[i] = jacobi1DPoisson(leftX, rightX, rhs, dx);
        }
        __syncthreads();
        double * tmp = x1; x1 = x0; x0 = tmp;
    }
  
}

__global__
void _jacobiUpdate(double * x1Gpu, const double * x0Gpu, const double * rhsGpu, const int nGrids, const int OVERLAP, const int subIterations)
{
    // Move to shared memory
    extern __shared__ double sharedMemory[];

    const int nPerSubdomain = blockDim.x + 2;

    // STEP 1 - MOVE ALL VALUES TO SHARED MEMORY
    const int I = threadIdx.x + (blockDim.x - OVERLAP) * blockIdx.x;
    const int i = threadIdx.x;
    if (I < nGrids) {
        sharedMemory[i] = x0Gpu[I];
        sharedMemory[i + nPerSubdomain] = x0Gpu[I];
        sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I]; // added
    }

    const int I2 = blockDim.x + (threadIdx.x + (blockDim.x - OVERLAP) * blockIdx.x);
    const int i2 = blockDim.x + threadIdx.x;
    if (i2 < nPerSubdomain && I2 < nGrids) { 
        sharedMemory[i2] = x0Gpu[I2];
        sharedMemory[i2 + nPerSubdomain] = x0Gpu[I2];
        sharedMemory[i2 + 2 * nPerSubdomain] = rhsGpu[I2]; // added
    }

    // Synchronize all threads before entering kernel
    __syncthreads();

    // STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK
    __jacobiUpdateKernel(nGrids, nPerSubdomain, OVERLAP, subIterations);

    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
    // Move back to global memory (if overlap = 0 ONLY)
/*    if ((I+1) < nGrids) {
        x0Gpu[I+1] = sharedMemory[i+1];
    } 
*/
    if ((I+1) < nGrids) {
        if (blockIdx.x == 0) {
            if (threadIdx.x <= blockDim.x - 1 - OVERLAP/2) {
                if ((subIterations % 2) == 0) {
                    x1Gpu[I+1] = sharedMemory[i+1];
                }
                else {
                    x1Gpu[I+1] = sharedMemory[i+1 + nPerSubdomain];
                }
            }
        }
        else if (blockIdx.x == gridDim.x - 1) {
            if (threadIdx.x >= OVERLAP/2) {
                if ((subIterations % 2) == 0) {
                    x1Gpu[I+1] = sharedMemory[i+1];
                }
                else {
                    x1Gpu[I+1] = sharedMemory[i+1 + nPerSubdomain];
                }
            }
        }
        else {
            if (threadIdx.x >= OVERLAP/2 && threadIdx.x <= blockDim.x - 1 - OVERLAP/2) {
                if ((subIterations % 2) == 0) {
                    x1Gpu[I+1] = sharedMemory[i+1];
                }
                else {
                    x1Gpu[I+1] = sharedMemory[i+1 + nPerSubdomain];
                }
            }
        } 
    }

    __syncthreads();
}


double * jacobiShared(const double * initX, const double * rhs, const int nGrids, const int cycles, const int threadsPerBlock, const int OVERLAP, const int subIterations)
{
    // Number of grid points handled by a subdomain
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((double)nGrids-2.0-(double)OVERLAP) / ((double)threadsPerBlock-(double)OVERLAP));

    // Allocate GPU memory via cudaMalloc
    double * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 3 * nSub * sizeof(double);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nGrids, OVERLAP, subIterations);
		{
            double * tmp = x1Gpu; x1Gpu = x0Gpu; x0Gpu = tmp;
		}
    }

    double * solution = new double[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiSharedIterationCountResidual(const double * initX, const double * rhs, const int nGrids, const double TOL, const int threadsPerBlock, const int OVERLAP, const int subIterations)
{
    // Number of grid points handled by a subdomain
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((double)nGrids-2.0-(double)OVERLAP) / ((double)threadsPerBlock-(double)OVERLAP));

    // Allocate GPU memory via cudaMalloc
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    cudaMalloc(&residualGpu, sizeof(double) * nGrids);

    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(residualGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 3 * nSub * sizeof(double);

    // Call kernel to allocate to sharedmemory and update points
    double residual = 1000000000000.0;
    int nIters = 0;
    double * residualCpu = new double[nGrids];
    while (residual > TOL) {
        
		// Perform one cycle of shared memory jacobi algorithm
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nGrids, OVERLAP, subIterations);
        // Perform CUDA error checking
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync)); 
        {
            double * tmp = x1Gpu; x1Gpu = x0Gpu; x0Gpu = tmp;
        }
        nIters++;
        
        // RESIDUAL
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST 
		residual1DPoissonGPU <<<numBlocks, threadsPerBlock>>> (residualGpu, x0Gpu, rhsGpu, nGrids);
        cudaDeviceSynchronize();
        cudaMemcpy(residualCpu, residualGpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
        residual = 0.0;
        for (int j = 0; j < nGrids; j++) {
            residual = residual + residualCpu[j];
        }
        residual = sqrt(residual);
#else
		cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
		residual = residual1DPoisson(solution, rhs, nGrids);
#endif
        // Print out the residual
		if (nIters % 10000 == 0) {
		    printf("Shared: The residual at step %d is %f\n", nIters, residual);
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

    return nIters;
}

int jacobiSharedIterationCountSolutionError(const double * initX, const double * rhs, const int nGrids, const double TOL, const int threadsPerBlock, const int OVERLAP, const int subIterations, const double * solution_exact)
{
    // Number of grid points handled by a subdomain
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((double)nGrids-2.0-(double)OVERLAP) / ((double)threadsPerBlock-(double)OVERLAP));

    // Allocate GPU memory via cudaMalloc
    double * x0Gpu, * x1Gpu, * rhsGpu, * solutionErrorGpu, * solution_exactGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    cudaMalloc(&solutionErrorGpu, sizeof(double) * nGrids);
    cudaMalloc(&solution_exactGpu, sizeof(double) * nGrids);

    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(solutionErrorGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(solution_exactGpu, solution_exact, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 3 * nSub * sizeof(double);

    // Call kernel to allocate to sharedmemory and update points
    double solution_error = 1000000000000.0;
    int nIters = 0;
    double * solutionErrorCpu = new double[nGrids];
    while (solution_error > TOL) {
        
        // Perform one cycle of shared memory jacobi algorithm
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nGrids, OVERLAP, subIterations);
        // Perform CUDA error checking
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync)); 
        {
            double * tmp = x1Gpu; x1Gpu = x0Gpu; x0Gpu = tmp;
        }
        nIters++;

        
        // ERROR CALCULATION
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST 
		solutionError1DPoissonGPU <<<numBlocks, threadsPerBlock>>> (solutionErrorGpu, x0Gpu, solution_exactGpu, nGrids);
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
        if (nIters % 1000 == 0) {
		    printf("Shared: The solution error at step %d is %f\n", nIters, solution_error);
        }
    }
 
    // Clean up
    // CPU
    delete[] solutionErrorCpu;
    // GPU
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(solutionErrorGpu);
    cudaFree(solution_exactGpu);

    return nIters;
}

