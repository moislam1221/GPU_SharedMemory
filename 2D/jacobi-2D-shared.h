#define GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST 1

__device__
void __jacobiUpdateKernel(const int nxGrids, const int nyGrids, const int subdomainLength_x, const int subdomainLength_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + subdomainLength_x * subdomainLength_y, * x2 = sharedMemory + 2 * subdomainLength_x * subdomainLength_y;

    const double dx = 1.0 / (nxGrids - 1);
    const double dy = 1.0 / (nyGrids - 1);

	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * subdomainLength_x;
   
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + ix;
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + iy;

    for (int k = 0; k < subIterations; k++) {
        // Compute local index in shared memory (indices constructed so only inner points updated)
		double leftX = x0[i-1];
		double rightX = x0[i+1];
		double topX = x0[i+subdomainLength_x];
		double bottomX = x0[i-subdomainLength_x];
		double rhs = x2[i];
 		if (Ix < nxGrids-1 && Iy < nyGrids-1) {
			x1[i] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs, dx, dy);
        }
	    __syncthreads();
        double * tmp = x0; x0 = x1; x1 = tmp;
    }
}

__global__
void _jacobiUpdate(double * x1Gpu, double * x0Gpu, const double * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    // Move to shared memory
    extern __shared__ double sharedMemory[];
   
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
    // While we haven't moved all points in global subdomain over to shared (some threads have to move multiple points - at least two)
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int i = sharedID; i < nPerSubdomain; i += stride) {
        // Compute the global ID of point in the grid
		Idx = (i % subdomainLength_x); // local x ID in subdomain
		Idy = i/subdomainLength_x; // local y ID in subdomain
		I = blockShift + Idx + Idy * nyGrids; // global ID 
        // If the global ID is less than number of points, or local ID is less than number of points in subdomain
        if (I < nDofs && i < nPerSubdomain) {
            sharedMemory[i] = x0Gpu[I]; 
            sharedMemory[i + nPerSubdomain] = x0Gpu[I];
            sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I];
        }
    }

    // Shift rhs array pointer based on blockIdx.x, blockIdx.y - not corrrect if rhsBlock is nonconstant
    // because when we move to next column in the subdomain we have to jump again in the rhs value
    // const double * rhsBlock = blockShift + rhsGpu;
    
    // Synchronize all threads before entering kernel
    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK
    __jacobiUpdateKernel(nxGrids, nyGrids, subdomainLength_x, subdomainLength_y, OVERLAP_X, OVERLAP_Y, subIterations);
	__syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
    // Move back to global memory (if overlap = 0 ONLY)
    // Local index of inner points
    // Define booleans indicating whether grid point should be handled by this block
    bool inxRange = 0;
    bool inyRange = 0;
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
        if ((subIterations % 2) == 0) { 
		    x1Gpu[I_inner] = sharedMemory[i_inner];
        }
        else { 
		    x1Gpu[I_inner] = sharedMemory[i_inner + nPerSubdomain];
        }
   
    }

    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

double * jacobiShared(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, const int cycles, const int threadsPerBlock_x, const int threadsPerBlock_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    // Number of grid points handled by a subdomain in each direction
    const int subdomainLength_x = threadsPerBlock_x + 2;
    const int subdomainLength_y = threadsPerBlock_y + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((double)nxGrids-2.0-(double)OVERLAP_X) / ((double)threadsPerBlock_x-(double)OVERLAP_X));
    const int nyBlocks = ceil(((double)nyGrids-2.0-(double)OVERLAP_Y) / ((double)threadsPerBlock_y-(double)OVERLAP_Y));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);

    // Define total number of degrees of freedom
    int nDofs = nxGrids * nyGrids;
    
    // Allocate GPU memory via cudaMalloc
    double * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
   
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 3 * subdomainLength_x * subdomainLength_y * sizeof(double);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, subIterations);
        {
            double * tmp = x1Gpu; x1Gpu = x0Gpu; x0Gpu = tmp;
        }
    }

    double * solution = new double[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiSharedIterationCountResidual(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, const double TOL, const int threadsPerBlock_x, const int threadsPerBlock_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations)
{
    // Number of grid points handled by a subdomain in each direction
    const int subdomainLength_x = threadsPerBlock_x + 2;
    const int subdomainLength_y = threadsPerBlock_y + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((double)nxGrids-2.0-(double)OVERLAP_X) / ((double)threadsPerBlock_x-(double)OVERLAP_X));
    const int nyBlocks = ceil(((double)nyGrids-2.0-(double)OVERLAP_Y) / ((double)threadsPerBlock_y-(double)OVERLAP_Y));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);

    // Define total number of degrees of freedom
    const int nDofs = nxGrids * nyGrids;

    // Allocate GPU memory via cudaMalloc
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&residualGpu, sizeof(double) * nDofs);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(residualGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

	// Container to hold CPU solution if one wants to compute residual purely on the CPU
#ifndef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
    double * solution = new double[nDofs];
#endif

    // Define amount of shared memory needed
    const int sharedBytes = 3 * subdomainLength_x * subdomainLength_y * sizeof(double);

    // Call kernel to allocate to sharedmemory and update points
    double residual = 1000000000000.0;
    int nIters = 0;
    double * residualCpu = new double[nDofs];
    while (residual > TOL) {
        
        // Perform one cycle of shared memory jacobi algorithm
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, subIterations);

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
        residual2DPoissonGPU <<<grid, block>>> (residualGpu, x0Gpu, rhsGpu, nxGrids, nyGrids);
        cudaDeviceSynchronize();
        cudaMemcpy(residualCpu, residualGpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
        residual = 0.0;
        for (int j = 0; j < nDofs; j++) {
            residual = residual + residualCpu[j];
        }
        residual = sqrt(residual);
#else
        cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
        residual = residual2DPoisson(solution, rhs, nxGrids, nyGrids);
#endif
        // Print out the residual
        if (nIters % 1000 == 0) {
			printf("The residual at step %d is %f\n", nIters, residual);
        }
    }

    // Clean up
    delete[] residualCpu;
#ifndef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
    delete[] solution;
#endif
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(residualGpu);

    return nIters;
}


int jacobiSharedIterationCountSolutionError(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, const double TOL, const int threadsPerBlock_x, const int threadsPerBlock_y, const int OVERLAP_X, const int OVERLAP_Y, const int subIterations, const double * solution_exact)
{
    // Number of grid points handled by a subdomain in each direction
    const int subdomainLength_x = threadsPerBlock_x + 2;
    const int subdomainLength_y = threadsPerBlock_y + 2;

    // Number of blocks necessary in each direction
    const int nxBlocks = ceil(((double)nxGrids-2.0-(double)OVERLAP_X) / ((double)threadsPerBlock_x-(double)OVERLAP_X));
    const int nyBlocks = ceil(((double)nyGrids-2.0-(double)OVERLAP_Y) / ((double)threadsPerBlock_y-(double)OVERLAP_Y));

    // Define the grid and block parameters
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);

    // Define total number of degrees of freedom
    const int nDofs = nxGrids * nyGrids;

    // Allocate GPU memory via cudaMalloc
    double * x0Gpu, * x1Gpu, * rhsGpu, * solutionErrorGpu, * solution_exactGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&solutionErrorGpu, sizeof(double) * nDofs);
    cudaMalloc(&solution_exactGpu, sizeof(double) * nDofs);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(solutionErrorGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(solution_exactGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

    // Container to hold CPU solution if one wants to compute residual purely on the CPU
#ifndef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
    double * solution = new double[nDofs];
#endif

    // Define amount of shared memory needed
    const int sharedBytes = 3 * subdomainLength_x * subdomainLength_y * sizeof(double);

    // Call kernel to allocate to sharedmemory and update points
    double solution_error = 1000000000000.0;
    int nIters = 0;
    double * solutionErrorCpu = new double[nDofs];
    while (solution_error > TOL) {
        
        // Perform one cycle of shared memory jacobi algorithm
        _jacobiUpdate <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, subIterations);

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

        // Error Calculation
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
        solutionError2DPoissonGPU <<<grid, block>>> (solutionErrorGpu, x0Gpu, solution_exactGpu, nDofs);
        cudaDeviceSynchronize();
        cudaMemcpy(solutionErrorCpu, solutionErrorGpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
        solution_error = 0.0;
        for (int j = 0; j < nDofs; j++) {
            solution_error = solution_error + solutionErrorCpu[j];
        }
        solution_error = sqrt(solution_error);
#else
        cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
        solution_error = solutionError2DPoisson(solution, solution_exact, nDofs);
#endif
        // Print out the residual
        if (nIters % 1000 == 0) {
			printf("The solution error at step %d is %f\n", nIters, solution_error);
        }
    }

    // Clean up
    delete[] solutionErrorCpu;
#ifndef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
    delete[] solution;
#endif
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(solutionErrorGpu);
    cudaFree(solution_exactGpu);

    return nIters;
}
