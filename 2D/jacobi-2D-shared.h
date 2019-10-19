__device__
void __jacobiUpdateKernel(const float * rhsBlock, const int nxGrids, const int nyGrids, const int nPerSubdomainEdge, const int OVERLAP, const int subIterations)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + nPerSubdomainEdge * nPerSubdomainEdge;

    const float dx = 1.0 / (nxGrids - 1);
    const float dy = 1.0 / (nyGrids - 1);

	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * nPerSubdomainEdge;
    for (int k = 0; k < subIterations; k++) {
        // Compute local index in shared memory (indices constructed so only inner points updated)
		float leftX = x0[i-1];
		float rightX = x0[i+1];
		float topX = x0[i+nPerSubdomainEdge];
		float bottomX = x0[i-nPerSubdomainEdge];
		x1[i] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhsBlock[i], dx, dy);
//        x1[i] = x0[i] + 1.0; //add(x0[i]);
        // printf("BlockIdx.x %d, BlockIdx.y %d: x1[%d] = %f\n", blockIdx.x, blockIdx.y, i, x1[i]);
        // Synchronize and swap points once all points are updated
        __syncthreads();
        float * tmp = x1; x1 = x0; x0 = tmp;
    }
}

__global__
void _jacobiUpdate(float * x0Gpu, const float * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP, const int subIterations)
{
    // Move to shared memory
    extern __shared__ float sharedMemory[];
    
    // Define useful constants regarding subdomain edge length and number of points within a 2D subdomain 
    const int nDofs = nxGrids * nyGrids;
    const int subdomainLength = (blockDim.x + 2);
    const int nPerSubdomain = subdomainLength * subdomainLength;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 1 - MOVE ALL VALUES TO SHARED MEMORY: Assume no overlap - OK
    // Start from 1D thread index and increment until all points in subdomain have been moved
	// Compute global ID of bottomleft corner point handled by specific blockIdx.x, blockIdx.y (serves as useful reference ID point)
    int blockShift = (blockIdx.x + blockIdx.y * nyGrids) * blockDim.x;
    int Idx, Idy, I;
    // While we haven't moved all points in global subdomain over to shared (some threads have to move multiple points - at least two)
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int i = sharedID; i < nPerSubdomain; i += stride) {
        // Compute the global ID of point in the grid
		Idx = (i % subdomainLength); // local x ID in subdomain
		Idy = i/subdomainLength; // local y ID in subdomain
		I = blockShift + Idx + Idy * nyGrids; // global ID 
        // printf("Block (%d, %d), Thread (%d, %d), SharedID %d: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, sharedID, i);
        /*if (blockIdx.x == 1 && blockIdx.y == 1) {
            printf("Block (%d, %d), global[%d] goes to shared[%d]\n", blockIdx.x, blockIdx.y, I, i);
        }*/
        // If the global ID is less than number of points, or local ID is less than number of points in subdomain
        if (I < nDofs && i < nPerSubdomain) {
            sharedMemory[i] = x0Gpu[I];
            sharedMemory[i + nPerSubdomain] = x0Gpu[I];
            // printf("Block: (%d, %d) sharedMemory[%d] = %f\n", blockIdx.x, blockIdx.y, i, sharedMemory[i]);
        }
    }
    // Shift rhs array pointer based on blockIdx.x, blockIdx.y - not corrrect if rhsBlock is nonconstant
    // because when we move to next column in the subdomain we have to jump again in the rhs value
    const float * rhsBlock = blockShift + rhsGpu;
    // Synchronize all threads before entering kernel
    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK
    __jacobiUpdateKernel(rhsBlock, nxGrids, nyGrids, subdomainLength, OVERLAP, subIterations);
    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
    // Move back to global memory (if overlap = 0 ONLY)
    // Local index of inner points
    const int i_inner = (threadIdx.x + 1) + (threadIdx.y + 1) * subdomainLength;
    const int Idx_inner = (i_inner % subdomainLength); // local ID
    const int Idy_inner = i_inner/subdomainLength; // local ID
    const int I_inner = blockShift + Idx_inner + Idy_inner * nyGrids; // global ID 
    x0Gpu[I_inner] = sharedMemory[i_inner];
    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

float * jacobiShared(const float * initX, const float * rhs, const int nxGrids, const int nyGrids, const int cycles, const int threadsPerBlock, const int OVERLAP, const int subIterations)
{
    // Number of grid points handled by a subdomain in each direction
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((float)nxGrids-2.0-(float)OVERLAP) / ((float)threadsPerBlock-(float)OVERLAP));
    const int nyBlocks = ceil(((float)nyGrids-2.0-(float)OVERLAP) / ((float)threadsPerBlock-(float)OVERLAP));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);

    // Define total number of degrees of freedom
    const int nDofs = nxGrids * nyGrids;

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * nSub * nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP, subIterations);
    }

    float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs,
            cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiSharedIterationCount(const float * initX, const float * rhs, const int nxGrids, const int nyGrids, const float TOL, const int threadsPerBlock, const int OVERLAP, const int subIterations)
{
    // Number of grid points handled by a subdomain in each direction
    const int nSub = threadsPerBlock + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((float)nxGrids-2.0-(float)OVERLAP) / ((float)threadsPerBlock-(float)OVERLAP));
    const int nyBlocks = ceil(((float)nyGrids-2.0-(float)OVERLAP) / ((float)threadsPerBlock-(float)OVERLAP));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);

    // Define total number of degrees of freedom
    const int nDofs = nxGrids * nyGrids;

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * nSub * nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    float residual = 1000000000000.0;
    int nIters = 0;
    float * solution = new float[nDofs];
    while (residual > TOL) {
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP, subIterations);
        nIters++;
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs, cudaMemcpyDeviceToHost);
        residual = residual2DPoisson(solution, rhs, nxGrids, nyGrids);
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
