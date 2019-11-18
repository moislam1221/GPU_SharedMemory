__device__
void __jacobiUpdateKernel(const float * rhsBlock, const int nxGrids, const int nyGrids, const int subdomainLength_x, const int subdomainLength_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + subdomainLength_x * subdomainLength_y;

    const float dx = 1.0 / (nxGrids - 1);
    const float dy = 1.0 / (nyGrids - 1);

	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * subdomainLength_x;
   
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + ix;
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + iy;

    for (int k = 0; k < subIterations; k++) {
        // Compute local index in shared memory (indices constructed so only inner points updated)
		float leftX = x0[i-1];
		float rightX = x0[i+1];
		float topX = x0[i+subdomainLength_x];
		float bottomX = x0[i-subdomainLength_x];
 		if (Ix < nxGrids-1 && Iy < nyGrids-1) {
			x1[i] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhsBlock[i], dx, dy);
            // x1[i] = x0[i] + 1.0;
        }
        
	    __syncthreads();
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
	
    __syncthreads();
}

__global__
void _jacobiUpdate(float * x0Gpu, const float * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    // Move to shared memory
    extern __shared__ float sharedMemory[];
   
    // Define useful constants regarding subdomain edge length and number of points within a 2D subdomain 
    const int nDofs = nxGrids * nyGrids;
    const int subdomainLength_x = blockDim.x + 2;
    const int subdomainLength_y = blockDim.y + 2;
    const int nPerSubdomain = subdomainLength_x * subdomainLength_y;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 1 - MOVE ALL VALUES TO SHARED MEMORY: Assume no overlap - OK
    // Start from 1D thread index and increment until all points in subdomain have been moved
	// Compute global ID of bottomleft corner point handled by specific blockIdx.x, blockIdx.y (serves as useful reference ID point)
    // int blockShift = (blockIdx.x + blockIdx.y * nyGrids) * blockDim.x;
    int blockShift = blockIdx.x * (blockDim.x - OVERLAP_X) + blockIdx.y * (blockDim.y - OVERLAP_Y) * nyGrids;
    int Idx, Idy, I;
    __syncthreads();
    // While we haven't moved all points in global subdomain over to shared (some threads have to move multiple points - at least two)
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    __syncthreads();
    for (int i = sharedID; i < nPerSubdomain; i += stride) {
        // Compute the global ID of point in the grid
		Idx = (i % subdomainLength_x); // local x ID in subdomain
		Idy = i/subdomainLength_x; // local y ID in subdomain
		I = blockShift + Idx + Idy * nyGrids; // global ID 
        // If the global ID is less than number of points, or local ID is less than number of points in subdomain
        if (I < nDofs && i < nPerSubdomain) {
            sharedMemory[i] = x0Gpu[I]; 
            sharedMemory[i + nPerSubdomain] = x0Gpu[I];
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
    __jacobiUpdateKernel(rhsBlock, nxGrids, nyGrids, subdomainLength_x, subdomainLength_y, OVERLAP_X, OVERLAP_Y, subIterations);
	__syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
    // Move back to global memory (if overlap = 0 ONLY)
    // Local index of inner points
    // Define booleans indicating whether grid point should be handled by this block
    bool inxRange = 0;
    bool inyRange = 0;
    __syncthreads();
    // Check if x point should be handled by this particular block
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + (threadIdx.x + 1);
    if (Ix < nxGrids-1){
		if (blockIdx.x == 0) {
			if (threadIdx.x <= blockDim.x - 1 - OVERLAP_X/2) {
				inxRange = 1;
			}
		}
		else if (blockIdx.x == gridDim.x - 1) {
			if (threadIdx.x >= OVERLAP_X/2) {
				inxRange = 1;
			}
		}
		else {
			if (threadIdx.x >= OVERLAP_X/2 && threadIdx.x <= blockDim.x - 1 - OVERLAP_X/2) {
				inxRange = 1;
			}
		}
    }
	__syncthreads();
    // Check if y point should be handled by this particular block
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + (threadIdx.y + 1);
    if (Iy < nyGrids-1) {
		if (blockIdx.y == 0) {
			if (threadIdx.y <= blockDim.y - 1 - OVERLAP_Y/2) {
				inyRange = 1;
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (threadIdx.y >= OVERLAP_Y/2) {
				inyRange = 1;
			}
		}
		else {
			if (threadIdx.y >= OVERLAP_Y/2 && threadIdx.y <= blockDim.y - 1 - OVERLAP_Y/2) {
				inyRange = 1;
			}
		}
    }
    __syncthreads();

    // If point is within bound of points to be handled by particular blockIdx.x, blockIdx.y, then move value over to global memory    
    if (inxRange == 1 && inyRange == 1) {
		const int i_inner = (threadIdx.x + 1) + (threadIdx.y + 1) * subdomainLength_x;
		const int Idx_inner = (i_inner % subdomainLength_x); // local ID
		const int Idy_inner = i_inner/subdomainLength_x; // local ID
		const int I_inner = blockShift + Idx_inner + Idy_inner * nyGrids; // global ID
        // printf("x0[%d] comes from blockIdx.x %d, blockIdx.y %d, sharedMemory[%d]\n", I_inner, blockIdx.x, blockIdx.y, i_inner); 
		x0Gpu[I_inner] = sharedMemory[i_inner];
    }

    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

float * jacobiShared(const float * initX, const float * rhs, const int nxGrids, const int nyGrids, const int cycles, const int threadsPerBlock_x, const int threadsPerBlock_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    // Number of grid points handled by a subdomain in each direction
    const int subdomainLength_x = threadsPerBlock_x + 2;
    const int subdomainLength_y = threadsPerBlock_y + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((float)nxGrids-2.0-(float)OVERLAP_X) / ((float)threadsPerBlock_x-(float)OVERLAP_X));
    const int nyBlocks = ceil(((float)nyGrids-2.0-(float)OVERLAP_Y) / ((float)threadsPerBlock_y-(float)OVERLAP_Y));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);

    // Define total number of degrees of freedom
    int nDofs = nxGrids * nyGrids;
    
    // Allocate GPU memory via cudaMalloc
    float * x0Gpu;
    float * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
   
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * subdomainLength_x * subdomainLength_y * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, subIterations);
    }

    float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiSharedIterationCount(const float * initX, const float * rhs, const int nxGrids, const int nyGrids, const float TOL, const int threadsPerBlock_x, const int threadsPerBlock_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    // Number of grid points handled by a subdomain in each direction
    const int subdomainLength_x = threadsPerBlock_x + 2;
    const int subdomainLength_y = threadsPerBlock_y + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((float)nxGrids-2.0-(float)OVERLAP_X) / ((float)threadsPerBlock_x-(float)OVERLAP_X));
    const int nyBlocks = ceil(((float)nyGrids-2.0-(float)OVERLAP_Y) / ((float)threadsPerBlock_y-(float)OVERLAP_Y));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);

    // Define total number of degrees of freedom
    const int nDofs = nxGrids * nyGrids;

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu;
    float * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * subdomainLength_x * subdomainLength_y * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    float residual = 1000000000000.0;
    int nIters = 0;
    float * solution = new float[nDofs];
    while (residual > TOL) {
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, subIterations);
        nIters++;
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs, cudaMemcpyDeviceToHost);
        residual = residual2DPoisson(solution, rhs, nxGrids, nyGrids);
        if (nIters % 10 == 0) {
            printf("Shared: The residual is %f\n", residual);
        }
    }

    // Clean up
    delete[] solution;
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);

    return nIters;
}
