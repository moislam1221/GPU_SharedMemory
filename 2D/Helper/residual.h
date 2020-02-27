__host__ __device__
double normFromRow(const double leftX, const double centerX, const double rightX, const double topX, const double bottomX, const double centerRhs, const double dx, const double dy)
{
    return centerRhs + (leftX + rightX) / (dx*dx) + (topX + bottomX) / (dy*dy) - (2.0/(dx*dx) + 2.0/(dy*dy)) * centerX;
}

double residual2DPoisson(const double * solution, const double * rhs, int nxGrids, int nyGrids)
{
    double residual = 0.0;
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    double leftX, centerX, rightX, topX, bottomX, residualContributionFromRow;

    int dof;
    for (int jGrid = 0; jGrid < nyGrids; jGrid++) {
		for (int iGrid = 0; iGrid < nxGrids; iGrid++) {
			if (iGrid == 0 || iGrid == nxGrids-1 || jGrid == 0 || jGrid == nyGrids-1) {
				residualContributionFromRow = 0.0;
			}
			else {
                dof = iGrid + jGrid * nxGrids;
				leftX = solution[dof - 1];
				centerX = solution[dof];
				rightX = solution[dof + 1];
				topX = solution[dof + nxGrids];
				bottomX = solution[dof - nxGrids];
				residualContributionFromRow = normFromRow(leftX, centerX, rightX, topX, bottomX, rhs[iGrid], dx, dy);
                // printf("CPU: The residual contribution for dof %d is %f\n", dof, residualContributionFromRow);
			}
			residual = residual + residualContributionFromRow * residualContributionFromRow;
		}
    }

    residual = sqrt(residual);
    return residual;
}

__global__
void residual2DPoissonGPU(double * residualGpu, const double * solution, const double * rhs, int nxGrids, int nyGrids)
{
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    double leftX, centerX, rightX, topX, bottomX, residualContributionFromRow;
    int nDofs = nxGrids * nyGrids;
 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * (blockDim.x * gridDim.x);        

    int iStride = blockDim.x * gridDim.x;
    int jStride = blockDim.y * gridDim.y;
    int Stride = iStride * jStride;

    for (int dof = index; dof < nDofs; dof = dof + Stride) { 
	    if (dof < nxGrids || dof >= nDofs - nxGrids) {
            residualContributionFromRow = 0.0;
        }
        else if ((dof % nxGrids == 0) || ((dof+1) % nxGrids == 0)) {
            residualContributionFromRow = 0.0;
        } 
        else {
			leftX = solution[dof - 1];
			centerX = solution[dof];
			rightX = solution[dof + 1];
			topX = solution[dof + nxGrids];
			bottomX = solution[dof - nxGrids];
			residualContributionFromRow = normFromRow(leftX, centerX, rightX, topX, bottomX, rhs[dof], dx, dy);
        }
		residualGpu[dof] = residualContributionFromRow * residualContributionFromRow;
    }
    __syncthreads();
}
