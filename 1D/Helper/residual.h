__host__ __device__
double normFromRow(const double leftX, const double centerX, const double rightX, const double centerRhs, const double dx)
{
    return centerRhs + (leftX - 2.0*centerX + rightX) / (dx*dx);
}

double residual1DPoisson(const double * solution, const double * rhs, int nGrids)
{
    double residual = 0.0;
    double dx = 1.0 / (nGrids - 1);
    double leftX, centerX, rightX, residualContributionFromRow;

    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
        if (iGrid == 0 || iGrid == nGrids-1) {
            residualContributionFromRow = 0.0;
        }
        else {
            leftX = solution[iGrid - 1];
            centerX = solution[iGrid];
            rightX = solution[iGrid + 1];
            residualContributionFromRow = normFromRow(leftX, centerX, rightX, rhs[iGrid], dx);
        }
        residual = residual + residualContributionFromRow * residualContributionFromRow;
	}

    residual = sqrt(residual);
    return residual;
}

__global__
void residual1DPoissonGPU(double * residualGpu, const double * solution, const double * rhs, int nGrids)
{
    double dx = 1.0 / (nGrids - 1);
    double leftX, centerX, rightX, residualContributionFromRow;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nGrids; i += stride) {
        if (i == 0 || i == nGrids-1) {
            residualContributionFromRow = 0.0;
        }
        else {
            leftX = solution[i - 1];
            centerX = solution[i];
            rightX = solution[i + 1];
            residualContributionFromRow = normFromRow(leftX, centerX, rightX, rhs[i], dx);
        }
        residualGpu[i] = residualContributionFromRow * residualContributionFromRow;
    }
    __syncthreads();
}

