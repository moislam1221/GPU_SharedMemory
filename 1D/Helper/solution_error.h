void loadSolutionExact(float * solution_exact, std::string SOLUTIONEXACT_FILENAME) 
{
    std::ifstream solution_exact_file(SOLUTIONEXACT_FILE_NAME);
    for (int i = 0; i < nGrids; i++) {
        solution_exact_file >> solution_exact[i];
    }
}


float solutionError1DPoisson(const float * solution, const float * solution_exact, int nGrids)
{
    float solutionError = 0.0;
    for (int i =0; i < nGrids; i++) {
        solutionError = solutionError + (solution_exact[i] - solution[i]) * (solution_exact[i] - solution[i]);
    }
    solutionError = sqrt(solutionError)
    
    return solutionError;
}

__global__
float solutionError1DPoissonGPU(float * solutionErrorGpu, const float * solution, const float * solution_exact, int nGrids)
{
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nGrids; i += stride) {
	solutionErrorGpu[i] = (solution_exact[i] - solution[i]) * (solution_exact[i] - solution[i]);
    }
    __syncthreads();


}

