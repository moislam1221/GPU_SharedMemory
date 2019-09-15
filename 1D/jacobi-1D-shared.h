__device__
void __jacobiUpdateKernel(const float * rhsBlock, const int nGrids, const int nSub, const int OVERLAP)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + nSub;

    const int numIterations = blockDim.x / 2;
    const float dx = 1.0 / (nGrids - 1);

    for (int k = 0; k < numIterations; k++) {
        int i = threadIdx.x + 1;
        int I = blockIdx.x * (blockDim.x - OVERLAP)+ i;
        if (I < nGrids-1) {
            float leftX = x0[i-1];
            float rightX = x0[i+1];
            x1[i] = jacobi1DPoisson(leftX, rightX, rhsBlock[i], dx);
        }
        __syncthreads();
        float * tmp = x1; x1 = x0; x0 = tmp;
    }
  
}

__global__
void _jacobiUpdate(float * x0Gpu, const float * rhsGpu, const int nGrids, const int OVERLAP)
{
    // Move to shared memory
    extern __shared__ float sharedMemory[];

    const int nPerSubdomain = blockDim.x + 2;

    // STEP 1 - MOVE ALL VALUES TO SHARED MEMORY
    const int I = threadIdx.x + (blockDim.x - OVERLAP) * blockIdx.x;
    const int i = threadIdx.x;
    if (I < nGrids) {
        sharedMemory[i] = x0Gpu[I];
        sharedMemory[i + nPerSubdomain] = x0Gpu[I];
    }

    const int I2 = blockDim.x + (threadIdx.x + (blockDim.x - OVERLAP) * blockIdx.x);
    const int i2 = blockDim.x + threadIdx.x;
    if (i2 < nPerSubdomain && I2 < nGrids) { 
        sharedMemory[i2] = x0Gpu[I2];
        sharedMemory[i2 + nPerSubdomain] = x0Gpu[I2];
    }

    const float * rhsBlock = rhsGpu + (blockDim.x - OVERLAP) * blockIdx.x;

    // Synchronize all threads before entering kernel
    __syncthreads();

    // STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK
    __jacobiUpdateKernel(rhsBlock, nGrids, nPerSubdomain, OVERLAP);


    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
    // Move back to global memory (if overlap = 0 ONLY)
/*    if ((I+1) < nGrids) {
        x0Gpu[I+1] = sharedMemory[i+1];
    } 
*/
    if ((I+1) < nGrids) {
        if (blockIdx.x == 0) {
            if (threadIdx.x <= blockDim.x - 1 - OVERLAP/2) {
                x0Gpu[I+1] = sharedMemory[i+1];
            }
        }
        else if (blockIdx.x == gridDim.x - 1) {
            if (threadIdx.x >= OVERLAP/2) {
                x0Gpu[I+1] = sharedMemory[i+1];
            }
        }
        else {
            if (threadIdx.x >= OVERLAP/2 && threadIdx.x <= blockDim.x - 1 - OVERLAP/2) {
                x0Gpu[I+1] = sharedMemory[i+1];
            }
        } 
    }

    __syncthreads();
}

float * jacobiShared(const float * initX, const float * rhs, const int nGrids, const int cycles, const int threadsPerBlock, const int OVERLAP)
{
    // Number of grid points handled by a subdomain
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0-(float)OVERLAP) / ((float)threadsPerBlock-(float)OVERLAP));

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, nGrids, OVERLAP);
    }

    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiSharedIterationCount(const float * initX, const float * rhs, const int nGrids, const int TOL, const int threadsPerBlock, const int OVERLAP)
{
    // Number of grid points handled by a subdomain
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0-(float)OVERLAP) / ((float)threadsPerBlock-(float)OVERLAP));

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    float residual = 100.0;
    int nIters = 0;
    float * solution = new float[nGrids];
    while (residual > TOL) {
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, nGrids, OVERLAP);
         nIters++;
         cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
         residual = residual1DPoisson(solution, rhs, nGrids);
         if (nIters % 100 == 0) {
             printf("Shared: The residual is %f\n", residual);
         }
    }

    // Clean up
    delete[] solution;
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return nIters;
}
