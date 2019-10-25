__device__
void __jacobiUpdateKernel(const float * rhsBlock, const int nGrids, const int nSub, const int OVERLAP, const int subIterations, const int operPerSubdomain)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + nSub;

    const float dx = 1.0 / (nGrids - 1);

    int I;
    for (int k = 0; k < subIterations; k++) {
        for (int i = threadIdx.x + 1; i < nSub - 1; i += blockDim.x) {
        // for (int idx = 0; idx < 1; idx++) {
        // int i = threadIdx.x + 1;    
            I = blockIdx.x * (operPerSubdomain - OVERLAP)+ i;
            if (I < nGrids-1) {
				float leftX = x0[i-1];
				float rightX = x0[i+1];
				x1[i] = jacobi1DPoisson(leftX, rightX, rhsBlock[i], dx);
				// x1[i] = x0[i] + 1.0;
			}
        }
        __syncthreads();
        float * tmp = x1; x1 = x0; x0 = tmp;
    }
}

__global__
void _jacobiUpdate(float * x0Gpu, const float * rhsGpu, const int nGrids, const int OVERLAP, const int subIterations, const int operPerSubdomain)
{
    // Move to shared memory
    extern __shared__ float sharedMemory[];

    const int nPerSubdomain = operPerSubdomain + 2;

    // STEP 1 - MOVE ALL VALUES TO SHARED MEMORY
    // recompute the shift for going to next block
    // make sure all threads move multiple poijnts properly
    int I;
    for (int i = threadIdx.x; i < nPerSubdomain; i += blockDim.x) {
		I = i + (operPerSubdomain - OVERLAP) * blockIdx.x;
		if (I < nGrids) {
			sharedMemory[i] = x0Gpu[I];
			sharedMemory[i + nPerSubdomain] = x0Gpu[I];
		}
    }

    const float * rhsBlock = rhsGpu + (operPerSubdomain - OVERLAP) * blockIdx.x;
    // Synchronize all threads before entering kernel
    __syncthreads();

    // STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK
    __jacobiUpdateKernel(rhsBlock, nGrids, nPerSubdomain, OVERLAP, subIterations, operPerSubdomain);

    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
    for (int i = threadIdx.x; i < nPerSubdomain; i += blockDim.x) {
		// int i = threadIdx.x;
        I = i + (operPerSubdomain - OVERLAP) * blockIdx.x;
        if (blockIdx.x == 0) {
            if (i <= operPerSubdomain - 1 - OVERLAP/2) {
                x0Gpu[I+1] = sharedMemory[i+1];
            }
        }
        else if (blockIdx.x == gridDim.x - 1) {
            if (i >= OVERLAP/2) { // ??
                x0Gpu[I+1] = sharedMemory[i+1];
            }
        }
        else {
            if (i >= OVERLAP/2 && i <= operPerSubdomain - 1 - OVERLAP/2) {
                x0Gpu[I+1] = sharedMemory[i+1];
            }
        }
    } 
    
    __syncthreads();
}

float * jacobiSharedLongerSubdomain(const float * initX, const float * rhs, const int nGrids, const int cycles, const int threadsPerBlock, const int OVERLAP, const int subIterations, const int innerSubdomainLength)
{
    // Number of grid points handled by a subdomain
    const int nSub = innerSubdomainLength + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0-(float)OVERLAP) / ((float)innerSubdomainLength-(float)OVERLAP));

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
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, nGrids, OVERLAP, subIterations, innerSubdomainLength);
    }

    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiSharedLongerSubdomainIterationCount(const float * initX, const float * rhs, const int nGrids, const float TOL, const int threadsPerBlock, const int OVERLAP, const int subIterations, const int innerSubdomainLength)
{
    // Number of grid points handled by a subdomain
    const int nSub = innerSubdomainLength + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0-(float)OVERLAP) / ((float)innerSubdomainLength-(float)OVERLAP));

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
    float residual = 1000000000000.0;
    int nIters = 0;
    float * solution = new float[nGrids];
    while (residual > TOL) {
        _jacobiUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, nGrids, OVERLAP, subIterations, innerSubdomainLength);
        nIters++;
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
        residual = residual1DPoisson(solution, rhs, nGrids);
        if (nIters % 1000 == 0) {
            printf("Shared: The residual is %f\n", residual);
        }
    }

    // Clean up
    delete[] solution;
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return nIters;
}
